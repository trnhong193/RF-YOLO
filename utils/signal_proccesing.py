import numpy as np
import torch
from scipy import signal

class SignalProcessor:
    """Các hàm xử lý tín hiệu RF"""
    
    @staticmethod
    def iq_to_spectrogram(iq_signal, fft_size=256, time_steps=256, 
                          overlap=0.5, window='hann'):
        """
        Chuyển đổi tín hiệu IQ thành spectrogram
        
        Args:
            iq_signal: Complex IQ samples (numpy array)
            fft_size: Kích thước FFT
            time_steps: Số time steps trong spectrogram
            overlap: Overlap ratio giữa các segments
            window: Window function
            
        Returns:
            spectrogram: (fft_size, time_steps) numpy array
        """
        # Tính hop length
        hop_length = int(fft_size * (1 - overlap))
        
        # Apply STFT
        f, t, Zxx = signal.stft(iq_signal, 
                                nperseg=fft_size,
                                noverlap=int(fft_size * overlap),
                                window=window)
        
        # Magnitude spectrogram
        spectrogram = np.abs(Zxx)
        
        # Resize to desired time_steps
        if spectrogram.shape[1] != time_steps:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, spectrogram.shape[1])
            x_new = np.linspace(0, 1, time_steps)
            
            spec_resized = np.zeros((fft_size, time_steps))
            for i in range(fft_size):
                f_interp = interp1d(x_old, spectrogram[i, :], 
                                   kind='linear', fill_value='extrapolate')
                spec_resized[i, :] = f_interp(x_new)
            spectrogram = spec_resized
        
        return spectrogram
    
    @staticmethod
    def normalize_spectrogram(spectrogram, method='log'):
        """
        Normalize spectrogram
        
        Args:
            spectrogram: Input spectrogram
            method: 'log', 'minmax', or 'standard'
        """
        if method == 'log':
            # Convert to dB scale
            spec_db = 20 * np.log10(spectrogram + 1e-10)
            # Normalize to [0, 1]
            spec_norm = (spec_db - spec_db.min()) / (
                spec_db.max() - spec_db.min() + 1e-10
            )
        elif method == 'minmax':
            spec_norm = (spectrogram - spectrogram.min()) / (
                spectrogram.max() - spectrogram.min() + 1e-10
            )
        elif method == 'standard':
            spec_norm = (spectrogram - spectrogram.mean()) / (
                spectrogram.std() + 1e-10
            )
        else:
            spec_norm = spectrogram
        
        return spec_norm
    
    @staticmethod
    def estimate_snr(iq_signal, signal_bandwidth, sample_rate):
        """
        Ước lượng SNR của tín hiệu
        
        Args:
            iq_signal: Complex IQ samples
            signal_bandwidth: Băng thông tín hiệu (Hz)
            sample_rate: Tần số lấy mẫu (Hz)
            
        Returns:
            snr_db: SNR in dB
        """
        # FFT
        fft_result = np.fft.fftshift(np.fft.fft(iq_signal))
        power_spectrum = np.abs(fft_result) ** 2
        
        # Tính bandwidth bins
        freq_resolution = sample_rate / len(iq_signal)
        bandwidth_bins = int(signal_bandwidth / freq_resolution)
        
        # Signal power (peak region)
        center = len(power_spectrum) // 2
        signal_power = np.mean(
            power_spectrum[center - bandwidth_bins//2:
                          center + bandwidth_bins//2]
        )
        
        # Noise power (từ edges)
        noise_power = np.mean(
            np.concatenate([
                power_spectrum[:bandwidth_bins],
                power_spectrum[-bandwidth_bins:]
            ])
        )
        
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    @staticmethod
    def detect_signal_regions(spectrogram, threshold_factor=2.0):
        """
        Phát hiện vùng có tín hiệu trong spectrogram
        
        Args:
            spectrogram: Input spectrogram
            threshold_factor: Ngưỡng = mean + threshold_factor * std
            
        Returns:
            mask: Binary mask của signal regions
        """
        # Compute threshold
        mean_power = np.mean(spectrogram)
        std_power = np.std(spectrogram)
        threshold = mean_power + threshold_factor * std_power
        
        # Binary mask
        mask = spectrogram > threshold
        
        # Morphological operations để làm sạch
        from scipy.ndimage import binary_opening, binary_closing
        mask = binary_closing(mask, structure=np.ones((3, 3)))
        mask = binary_opening(mask, structure=np.ones((3, 3)))
        
        return mask.astype(float)

class RealTimeProcessor:
    """Xử lý real-time cho inference"""
    
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.FFT_SIZE * config.TIME_STEPS
        self.buffer = np.zeros(self.buffer_size, dtype=complex)
        self.buffer_idx = 0
        
    def add_samples(self, new_samples):
        """
        Thêm samples mới vào buffer
        
        Args:
            new_samples: Complex IQ samples
            
        Returns:
            spectrograms: List of spectrograms ready for inference
        """
        spectrograms = []
        
        for sample in new_samples:
            self.buffer[self.buffer_idx] = sample
            self.buffer_idx += 1
            
            # Khi buffer đầy
            if self.buffer_idx >= self.buffer_size:
                # Tạo spectrogram
                processor = SignalProcessor()
                spec = processor.iq_to_spectrogram(
                    self.buffer,
                    fft_size=self.config.FFT_SIZE,
                    time_steps=self.config.TIME_STEPS
                )
                spec = processor.normalize_spectrogram(spec, method='log')
                spectrograms.append(spec)
                
                # Shift buffer (overlap)
                overlap_samples = self.buffer_size // 2
                self.buffer[:overlap_samples] = self.buffer[overlap_samples:]
                self.buffer_idx = overlap_samples
        
        return spectrograms