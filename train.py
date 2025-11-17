import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from config import Config
from data.dataset import DroneDataset
from models.yolo_lite import YOLOLite
from models.loss import YOLOLoss
from utils.visualization import Visualizer

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Create directories
        Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model = YOLOLite(config).to(self.device)
        print(f"Model created with {self.count_parameters()} parameters")
        
        # Loss
        self.criterion = YOLOLoss(config)
        
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), 
                            lr=config.LEARNING_RATE,
                            weight_decay=1e-5)
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                          mode='min',
                                          factor=0.5,
                                          patience=5,
                                          verbose=True)
        
        # Data loaders
        self.train_loader = self.create_dataloader('train')
        self.val_loader = self.create_dataloader('val')
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_loc_loss': [],
            'val_loc_loss': [],
            'train_conf_loss': [],
            'val_conf_loss': [],
            'train_class_loss': [],
            'val_class_loss': []
        }
        
        self.best_val_loss = float('inf')
        self.visualizer = Visualizer(config)
    
    def create_dataloader(self, split):
        dataset = DroneDataset(self.config, split=split)
        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
        return loader
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        total_loc_loss = 0
        total_conf_loss = 0
        total_class_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (spectrograms, targets) in enumerate(pbar):
            spectrograms = spectrograms.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(spectrograms)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_loc_loss += loss_dict['localization'].item()
            total_conf_loss += loss_dict['confidence'].item()
            total_class_loss += loss_dict['classification'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'loc': f"{loss_dict['localization'].item():.4f}",
                'conf': f"{loss_dict['confidence'].item():.4f}",
                'cls': f"{loss_dict['classification'].item():.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_loc = total_loc_loss / len(self.train_loader)
        avg_conf = total_conf_loss / len(self.train_loader)
        avg_class = total_class_loss / len(self.train_loader)
        
        return avg_loss, avg_loc, avg_conf, avg_class
    
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        total_loc_loss = 0
        total_conf_loss = 0
        total_class_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for spectrograms, targets in pbar:
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)
                
                # Forward
                predictions = self.model(spectrograms)
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss_dict['total'].item()
                total_loc_loss += loss_dict['localization'].item()
                total_conf_loss += loss_dict['confidence'].item()
                total_class_loss += loss_dict['classification'].item()
                
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}"
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_loc = total_loc_loss / len(self.val_loader)
        avg_conf = total_conf_loss / len(self.val_loader)
        avg_class = total_class_loss / len(self.val_loader)
        
        return avg_loss, avg_loc, avg_conf, avg_class
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save last checkpoint
        last_path = Path(self.config.CHECKPOINT_DIR) / 'last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.CHECKPOINT_DIR) / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']
    
    def train(self, num_epochs=None, resume_from=None):
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        start_epoch = 0
        
        # Resume training
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_loc, train_conf, train_class = self.train_epoch()
            
            # Validate
            val_loss, val_loc, val_conf, val_class = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Loc: {train_loc:.4f}, "
                  f"Conf: {train_conf:.4f}, Cls: {train_class:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Loc: {val_loc:.4f}, "
                  f"Conf: {val_conf:.4f}, Cls: {val_class:.4f})")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_loc_loss'].append(train_loc)
            self.history['val_loc_loss'].append(val_loc)
            self.history['train_conf_loss'].append(train_conf)
            self.history['val_conf_loss'].append(val_conf)
            self.history['train_class_loss'].append(train_class)
            self.history['val_class_loss'].append(val_class)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Save history
            history_path = Path(self.config.LOG_DIR) / 'history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        # Plot training history
        self.visualizer.plot_training_history(
            self.history,
            save_path=Path(self.config.LOG_DIR) / 'training_history.png'
        )

def main():
    config = Config()
    trainer = Trainer(config)
    
    # Train
    trainer.train(num_epochs=config.NUM_EPOCHS)

if __name__ == '__main__':
    main()