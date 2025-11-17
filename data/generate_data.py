import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import json

class DroneSignalGenerator:
    """Tạo tín hiệu drone mô phỏng"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.fft_size = config.FFT_SIZE
        self.time_steps = config.TIME_STEPS
        
    def generate_fhss_signal(self, duration, hop_rate, num_hops, bandwidth):
        """
        Tạo tín hiệu FHSS (Frequency Hopping Spread Spectrum)
        Dùng cho RC (Remote Control) signal của drone
        """
        num_samples = int(duration * self.sample_rate)
        signal = np.zeros(num_samples, dtype=complex)
        
        samples_per_hop = int(self.sample_rate / hop_rate)
        
        # Tạo các hop frequencies ngẫu nhiên
        hop_freqs = np.random.uniform(-bandwidth/2, bandwidth/2, num_hops)
        
        for i in range(num_hops):
            start_idx = i * samples_per_hop
            end_idx = min(start_idx + samples_per_hop, num_samples)
            
            if start_idx >= num_samples:
                break
                
            # Tạo tone tại frequency hop
            t = np.arange(end_idx - start_idx) / self.sample_rate
            freq = hop_freqs[i % len(hop_freqs)]
            
            # GFSK modulation
            phase = 2 * np.pi * freq * t
            signal[start_idx:end_idx] = np.exp(1j * phase)
            
        return signal
    
    def generate_dsss_signal(self, duration, center_freq, bandwidth, spreading_factor=10):
        """
        Tạo tín hiệu DSSS (Direct Sequence Spread Spectrum)  
        Dùng cho video transmission của drone
        """
        num_samples = int(duration * self.sample_rate)
        
        # Chip rate
        chip_rate = bandwidth * spreading_factor
        samples_per_chip = int(self.sample_rate / chip_rate)
        
        # PN sequence
        num_chips = num_samples // samples_per_chip
        pn_sequence = np.random.choice([-1, 1], num_chips)
        
        # Upsample và tạo baseband signal
        signal = np.repeat(pn_sequence, samples_per_chip)[:num_samples]
        
        # Modulate lên carrier
        t = np.arange(num_samples) / self.sample_rate
        carrier = np.exp(2j * np.pi * center_freq * t)
        
        signal = signal * carrier
        
        return signal.astype(complex)
    
    def generate_wifi_signal(self, duration, center_freq, bandwidth=20e6):
        """Tạo tín hiệu WiFi (OFDM-like)"""
        num_samples = int(duration * self.sample_rate)
        
        # OFDM subcarriers
        num_subcarriers = 52  # WiFi 802.11
        subcarrier_spacing = bandwidth / num_subcarriers
        
        signal = np.zeros(num_samples, dtype=complex)
        t = np.arange(num_samples) / self.sample_rate
        
        for i in range(num_subcarriers):
            freq = center_freq + (i - num_subcarriers/2) * subcarrier_spacing
            # Random QAM symbols
            data = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], 
                                   size=num_samples)
            carrier = np.exp(2j * np.pi * freq * t)
            signal += data * carrier
            
        return signal / np.sqrt(num_subcarriers)
    
    def add_awgn(self, signal, snr_db):
        """Thêm nhiễu AWGN"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(len(signal)) + 
            1j*np.random.randn(len(signal))
        )
        
        return signal + noise
    
    def generate_spectrogram(self, signal):
        """Chuyển IQ signal thành spectrogram"""
        # Reshape signal
        num_segments = self.time_steps
        segment_length = len(signal) // num_segments
        
        spectrogram = np.zeros((self.fft_size, num_segments))
        
        for i in range(num_segments):
            start = i * segment_length
            end = start + self.fft_size
            
            if end > len(signal):
                segment = np.pad(signal[start:], 
                               (0, end - len(signal)), 
                               mode='constant')
            else:
                segment = signal[start:end]
            
            # FFT
            fft_result = np.fft.fftshift(np.fft.fft(segment))
            spectrogram[:, i] = np.abs(fft_result)
        
        # Normalize
        spectrogram = 20 * np.log10(spectrogram + 1e-10)
        spectrogram = (spectrogram - spectrogram.min()) / (
            spectrogram.max() - spectrogram.min() + 1e-10
        )
        
        return spectrogram
    
    def generate_sample(self, num_signals=None, snr_range=(-10, 10)):
        """
        Tạo một sample training với multiple signals
        
        Returns:
            spectrogram: (256, 256) numpy array
            annotations: list of dicts with bounding boxes and classes
        """
        if num_signals is None:
            num_signals = np.random.randint(1, 5)  # 1-4 signals
        
        duration = self.time_steps * self.fft_size / self.sample_rate
        total_samples = self.time_steps * self.fft_size
        
        combined_signal = np.zeros(total_samples, dtype=complex)
        annotations = []
        
        for _ in range(num_signals):
            # Random class
            class_idx = np.random.randint(0, len(self.config.CLASSES) - 1)
            class_name = self.config.CLASSES[class_idx]
            
            # Random timing
            start_time = np.random.uniform(0, duration * 0.7)
            signal_duration = np.random.uniform(duration * 0.2, duration * 0.5)
            
            # Random frequency position
            freq_offset = np.random.uniform(-0.4, 0.4)  # Normalized
            
            if 'DJI' in class_name or 'Parrot' in class_name:
                # Drone RC signal (FHSS)
                bandwidth = np.random.uniform(1e6, 5e6)
                hop_rate = np.random.uniform(50, 200)
                num_hops = int(signal_duration * hop_rate)
                
                signal = self.generate_fhss_signal(
                    signal_duration, hop_rate, num_hops, bandwidth
                )
                
            elif 'WiFi' in class_name:
                signal = self.generate_wifi_signal(
                    signal_duration, 
                    freq_offset * self.sample_rate / 2,
                    bandwidth=20e6
                )
                
            else:
                # DSSS for video
                bandwidth = np.random.uniform(10e6, 40e6)
                signal = self.generate_dsss_signal(
                    signal_duration,
                    freq_offset * self.sample_rate / 2,
                    bandwidth
                )
            
            # Add noise
            snr = np.random.uniform(*snr_range)
            signal = self.add_awgn(signal, snr)
            
            # Place signal in time
            start_sample = int(start_time * self.sample_rate)
            end_sample = min(start_sample + len(signal), total_samples)
            signal_length = end_sample - start_sample
            
            combined_signal[start_sample:end_sample] += signal[:signal_length]
            
            # Create annotation (normalized coordinates)
            # x_center, y_center, width, height in [0, 1]
            x_center = (start_time + signal_duration/2) / duration
            y_center = (freq_offset + 1) / 2  # Convert [-0.5, 0.5] to [0, 1]
            width = signal_duration / duration
            height = (bandwidth / self.sample_rate)
            
            annotations.append({
                'class': class_idx,
                'class_name': class_name,
                'bbox': [x_center, y_center, width, height]
            })
        
        # Generate spectrogram
        spectrogram = self.generate_spectrogram(combined_signal)
        
        return spectrogram, annotations

def generate_dataset(config, num_train=5000, num_val=1000, num_test=500):
    """Tạo toàn bộ dataset"""
    
    generator = DroneSignalGenerator(config)
    output_dir = Path(config.DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_split(split_name, num_samples):
        spectrograms = []
        all_annotations = []
        
        print(f"\nGenerating {split_name} split...")
        for i in tqdm(range(num_samples)):
            spec, annots = generator.generate_sample()
            spectrograms.append(spec)
            all_annotations.append(annots)
        
        # Save to HDF5
        h5_path = output_dir / f'{split_name}.h5'
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('spectrograms', 
                           data=np.array(spectrograms).astype(np.float32))
        
        # Save annotations to JSON
        json_path = output_dir / f'{split_name}_annotations.json'
        with open(json_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        print(f"Saved {num_samples} samples to {h5_path}")
    
    save_split('train', num_train)
    save_split('val', num_val)
    save_split('test', num_test)
    
    # Save class names
    class_path = output_dir / 'classes.json'
    with open(class_path, 'w') as f:
        json.dump(config.CLASSES, f, indent=2)

if __name__ == '__main__':
    from config import Config
    config = Config()
    generate_dataset(config, num_train=5000, num_val=1000, num_test=500)