# data loader - IMPROVED VERSION
from __future__ import print_function, division
from email.mime import image
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
#==========================dataset load==========================

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label = sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label = sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl}

class CenterCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		# print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
		assert((h >= new_h) and (w >= new_w))

		h_offset = int(math.floor((h - new_h)/2))
		w_offset = int(math.floor((w - new_w)/2))

		image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
		label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

		return {'image': image, 'label': label}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h + 1)
		left = np.random.randint(0, w - new_w + 1)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'image': image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 1. Ép kiểu và chuẩn hóa về [0, 1] ĐỒNG NHẤT (luôn chia 255)
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0

        # 2. Xử lý ảnh xám (1 channel) thành 3 channels nếu cần
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # 3. Normalize theo thông số chuẩn của ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # 4. Xử lý Label: Đảm bảo có 3 chiều (C, H, W) và kẹp giá trị [0, 1]
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        label = np.clip(label, 0, 1) # Chống lỗi Assertion trên GPU

        # 5. Chuyển từ HWC (Numpy) sang CHW (PyTorch)
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float()
        }

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, label = sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = tmpLbl.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg),
			'label': torch.from_numpy(tmpLbl)}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image = io.imread(self.image_name_list[idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		#print("len of label3")
		#print(len(label_3.shape))
		#print(label_3.shape)

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		# #vertical flipping
		# # fliph = np.random.randn(1)
		# flipv = np.random.randn(1)
		#
		# if flipv>0:
		# 	image = image[::-1,:,:]
		# 	label = label[::-1,:,:]
		# #vertical flip

		sample = {'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample

# ==================== AUGMENTATION CLASSES ====================
# These classes help prevent overfitting by adding variety to training data

class RandomHorizontalFlip(object):
	"""
	Randomly flip image and label horizontally
	Args:
		p (float): probability of flipping (default: 0.5)
	"""
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		if random.random() < self.p:
			image = np.fliplr(image).copy()
			label = np.fliplr(label).copy()

		return {'image': image, 'label': label}

class ColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        image, label = sample['image'], sample['label']
        
        # Ép kiểu sang float32 để tính toán chính xác, tránh overflow
        image = image.astype(np.float32)

        # 1. Random brightness (Độ sáng)
        if self.brightness > 0 and random.random() > 0.5:
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            image *= alpha

        # 2. Random contrast (Độ tương phản)
        if self.contrast > 0 and random.random() > 0.5:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            gray = image.mean()
            image = (image - gray) * alpha + gray

        # 3. Random saturation (Độ bão hòa - Bổ sung phần bạn đang thiếu)
        if self.saturation > 0 and random.random() > 0.5:
            alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
            # Tính toán kênh xám dựa trên trọng số chuẩn của mắt người
            gray_img = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
            image = (image - gray_img) * alpha + gray_img

        # Luôn kẹp giá trị về [0, 255] sau khi biến đổi màu
        image = np.clip(image, 0, 255)

        return {'image': image, 'label': label}

class GaussianBlur(object):
	"""
	Randomly apply Gaussian blur to image
	Prevents overfitting to high-frequency details
	Args:
		kernel_size (int): size of blur kernel (default: 5)
		p (float): probability of applying blur (default: 0.3)
	"""
	def __init__(self, kernel_size=5, p=0.3):
		self.kernel_size = kernel_size
		self.p = p

	def __call__(self, sample):
		if random.random() > self.p:
			return sample

		image, label = sample['image'], sample['label']

		try:
			from scipy.ndimage import gaussian_filter
			sigma = random.uniform(0.1, 2.0)
			# Apply blur only to image, not label
			image = gaussian_filter(image, sigma=(sigma, sigma, 0))
		except ImportError:
			# If scipy not available, skip blur
			pass

		return {'image': image, 'label': label}

class RandomRotation(object):
	"""
	Randomly rotate image and label by small angle
	Helps model learn rotational invariance
	Args:
		degrees (float): max rotation angle (default: 15)
		p (float): probability of rotation (default: 0.5)
	"""
	def __init__(self, degrees=15, p=0.5):
		self.degrees = degrees
		self.p = p

	def __call__(self, sample):
		if random.random() > self.p:
			return sample

		image, label = sample['image'], sample['label']

		try:
			from scipy.ndimage import rotate
			angle = random.uniform(-self.degrees, self.degrees)
			# Rotate both image and label with same angle
			image = rotate(image, angle, reshape=False, order=1, mode='nearest')
			label = rotate(label, angle, reshape=False, order=0, mode='nearest')
		except ImportError:
			# If scipy not available, skip rotation
			pass

		return {'image': image, 'label': label}

class GaussianNoise(object):
    def __init__(self, mean=0, std=0.01, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        image, label = sample['image'], sample['label']

        # 1. Đảm bảo image là float để cộng không bị tràn số hoặc lỗi dtype
        image = image.astype(np.float32)
        
        # 2. Tạo noise cùng shape với image
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        
        # 3. Cộng noise (nhân 255 vì std ông để 0.01 là theo dải 0-1)
        # Nếu image của ông đã chuẩn hóa về 0-1 rồi thì bỏ * 255 đi
        image = image + noise * 255.0
        
        # 4. Clip để giữ giá trị trong dải pixel hợp lệ
        image = np.clip(image, 0, 255)

        return {'image': image, 'label': label}
	
class RandomVerticalFlip(object):
	"""
	Randomly flip image and label vertically
	Args:
		p (float): probability of flipping (default: 0.5)
	"""
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		if random.random() < self.p:
			image = np.flipud(image).copy()
			label = np.flipud(label).copy()

		return {'image': image, 'label': label}
