import numpy as np
from typing import Iterator, List, Optional, Callable, Tuple, Any
from pathlib import Path
import random


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement __getitem__")


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("At least one tensor must be provided")
        
        # Check all tensors have same length
        length = len(tensors[0])
        if not all(len(t) == length for t in tensors):
            raise ValueError("All tensors must have the same size in the first dimension")
        
        self.tensors = tensors
    
    def __len__(self) -> int:
        return len(self.tensors[0])
    
    def __getitem__(self, idx: int):
        return tuple(tensor[idx] for tensor in self.tensors)


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator:
        # Generate indices
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Skip last incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Fetch samples
            batch = [self.dataset[idx] for idx in batch_indices]
            
            # Collate into batch
            yield self.collate_fn(batch)


def default_collate(batch: List[Any]) -> Tuple:
    if not batch:
        return tuple()
    
    # Get first element to determine structure
    elem = batch[0]
    
    if isinstance(elem, tuple):
        # Recursively collate each element of the tuple
        return tuple(default_collate([sample[i] for sample in batch]) 
                    for i in range(len(elem)))
    
    elif isinstance(elem, np.ndarray):
        # Stack numpy arrays
        return np.stack(batch, axis=0)
    
    elif isinstance(elem, (int, float)):
        # Convert scalars to array
        return np.array(batch)
    
    elif hasattr(elem, 'data'):  # PySML Tensor
        # Stack tensor data
        return np.stack([item.data for item in batch], axis=0)
    
    else:
        # Return as list for other types
        return batch


def random_split(dataset: Dataset, lengths: List[int], seed: Optional[int] = None) -> List[Subset]:
    if sum(lengths) != len(dataset):
        raise ValueError(f"Sum of lengths ({sum(lengths)}) must equal dataset length ({len(dataset)})")
    
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate random indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Split indices
    subsets = []
    start = 0
    for length in lengths:
        end = start + length
        subsets.append(Subset(dataset, indices[start:end]))
        start = end
    
    return subsets


def train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[Subset, Subset]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    n_test = int(len(dataset) * test_size)
    n_train = len(dataset) - n_test
    
    return random_split(dataset, [n_train, n_test], seed=random_state)


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions
        
        # Build class to index mapping
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Find all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), class_idx))
        
        if not self.samples:
            raise RuntimeError(f"No images found in {root}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        img_path, label = self.samples[idx]
        
        # Load image (requires PIL or opencv)
        try:
            from PIL import Image
            img = np.array(Image.open(img_path))
        except ImportError:
            try:
                import cv2
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ImportError:
                raise RuntimeError(
                    "ImageFolderDataset requires either PIL (pip install Pillow) "
                    "or OpenCV (pip install opencv-python)"
                )
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class CSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        skip_header: bool = True
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        
        # Load CSV data
        data = np.genfromtxt(
            self.csv_path,
            delimiter=',',
            skip_header=1 if skip_header else 0,
            names=True
        )
        
        # Extract features and targets
        if target_column:
            self.targets = data[target_column]
            
            if feature_columns:
                self.features = np.column_stack([data[col] for col in feature_columns])
            else:
                # Use all columns except target
                all_cols = data.dtype.names
                feature_cols = [col for col in all_cols if col != target_column]
                self.features = np.column_stack([data[col] for col in feature_cols])
        else:
            self.features = data
            self.targets = None
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int):
        features = self.features[idx]
        
        if self.transform:
            features = self.transform(features)
        
        if self.targets is not None:
            return features, self.targets[idx]
        else:
            return features


class BatchSampler:
    def __init__(self, sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# Common transforms
class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, data):
        return (data - self.mean) / self.std


class ToTensor:
    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data.astype(np.float32)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return np.fliplr(img)
        return img


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(f"Image size {(h, w)} is smaller than crop size {self.size}")
        
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        
        return img[top:top+th, left:left+tw]
