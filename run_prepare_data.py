
import lmdb
import argparse
import multiprocessing

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from functools import partial

from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(path,
                             max_readers=32,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        return img


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()
    return val


def resize_multiple(img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100):
    imgs = [resize_and_convert(img, size, resample, quality) for size in sizes]
    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file).convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)
    return i, out


def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument("--size", type=str, default="128,256,512,1024", help="resolutions of images for the dataset")
    parser.add_argument("--n_worker", type=int, default=8, help="number of workers for preparing dataset")
    parser.add_argument("--resample", type=str, default="bicubic", help="resampling methods for resizing images")
    parser.add_argument("path", type=str, help="path to the image dataset")
    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
