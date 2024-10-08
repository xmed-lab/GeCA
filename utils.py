import os
import io
import base64
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import requests
import itertools
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import fold, unfold
from einops import rearrange, repeat
import math
import random
from scipy import signal

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
# from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

"""
CPU-based utility functions
"""

def pair(t):
	return t if isinstance(t, tuple) else (t, t)

def imread(url, max_size=None, mode=None):
	if url.startswith(('http:', 'https:')):
		r = requests.get(url)
		f = io.BytesIO(r.content)
	else:
		f = url
	img = PIL.Image.open(f)
	if max_size is not None:
		img.thumbnail((max_size[0], max_size[1]), PIL.Image.ANTIALIAS)
	if mode is not None:
		img = img.convert(mode)
	img = np.float32(img)/255.0
	return img

def py2pil(a):
	return np2pil(py2np(a))

def np2pil(a):
	if a.dtype in [np.float32, np.float64]:
		a = np.uint8(np.clip(a, 0, 1)*255)
	return PIL.Image.fromarray(a, mode=guess_mode(a))

def py2np(a):
	if a.shape[-1] in [1, 3, 4]:
		return a.detach().cpu().numpy()
	if len(a.shape) == 4:
		return a.permute(0, 2, 3, 1).detach().cpu().numpy()
	elif len(a.shape) == 3:
		return a.permute(1, 2, 0).detach().cpu().numpy()
	else:
		return a.detach().cpu().numpy()

def guess_mode(data):
	if data.shape[-1] == 1:
		return 'L'
	if data.shape[-1] == 3:
		return 'RGB'
	if data.shape[-1] == 4:
		return 'RGBA'
	raise ValueError('Un-supported shape for image conversion %s' % list(data.shape))

def imwrite(f, img, fmt=None):
	if torch.is_tensor(img):
		img = py2np(img)
	assert len(img.shape) < 4, 'batch dim not supported'  # no batch dim allowed
	if len(img.shape) == 2:
		img = np.repeat(img[..., None], 3, -1)
	elif img.shape[-1] == 1:
		img = np.repeat(img, 3, -1)
	elif img.shape[-1] == 4:
		img = img[..., :3]
	if isinstance(f, str):
		fmt = f.rsplit('.', 1)[-1].lower()
		if fmt == 'jpg':
			fmt = 'jpeg'
		f = open(f, 'wb')
	np2pil(img).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
	img = np.asarray(img)
	if len(img.shape) == 3 and img.shape[-1] == 4:
		fmt = 'png'
	f = io.BytesIO()
	imwrite(f, img, fmt)
	return f.getvalue()

def im2url(a, fmt='jpeg'):
	encoded = imencode(a, fmt)
	base64_byte_string = base64.b64encode(encoded).decode('ascii')
	return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def tile2d(a, w=None):
	a = np.asarray(a)
	if w is None:
		w = int(np.ceil(np.sqrt(len(a))))
	th, tw = a.shape[1:3]
	pad = (w-len(a))%w
	a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
	h = len(a)//w
	a = a.reshape([h, w]+list(a.shape[1:]))
	a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
	return a

def zoom(img, scale=4):
	img = np.repeat(img, scale, 0)
	img = np.repeat(img, scale, 1)
	return img

def plot(imgs, col_title=None, row_title=None, imshow_kwargs=None):
	if not isinstance(imgs[0], list):
		# Make a 2d grid even if there's just 1 row
		imgs = [imgs]

	num_rows = len(imgs)
	num_cols = len(imgs[0])
	fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
	for row_idx, row in enumerate(imgs):
		for col_idx, img in enumerate(row):
			ax = axs[row_idx, col_idx]
			try:
				ax.imshow(np.asarray(img), **imshow_kwargs[row_idx][col_idx])
			except:
				ax.imshow(np.asarray(img))
			ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	if col_title is not None:
		for col_idx in range(num_cols):
			axs[0, col_idx].set(xlabel=col_title[col_idx])
	if row_title is not None:
		for row_idx in range(num_rows):
			axs[row_idx, 0].set(ylabel=row_title[row_idx])

	plt.tight_layout()

def deq_log_plot(f_trace, b_trace):
	# create figure
	fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, dpi=150)

	# Create a subplot
	axs[0,0].semilogy(np.arange(1, len(f_trace['abs_trace'])+1), f_trace['abs_trace'])
	axs[0,0].semilogy(np.arange(1, len(b_trace['abs_trace'])+1), b_trace['abs_trace'])
	axs[0,1].semilogy(np.arange(1, len(f_trace['rel_trace'])+1), f_trace['rel_trace'])
	axs[0,1].semilogy(np.arange(1, len(b_trace['rel_trace'])+1), b_trace['rel_trace'])

	# create legend
	axs[0,0].legend(['Forward', 'Backward'])
	axs[0,1].legend(['Forward', 'Backward'])

	axs[0,0].set(xlabel='Iteration', ylabel='Residual', title='abs diff')
	axs[0,1].set(xlabel='Iteration', ylabel='Residual', title='rel diff')

	fig.set_tight_layout(True)

	return fig

def arr_from_fig(fig, output_shape, dpi, transparency=False):
	buf = io.BytesIO()
	fig.savefig(buf, format='raw', dpi=dpi)
	buf.seek(0)
	img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
						 newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
	plt.close(fig)
	if not transparency:
		img_arr = img_arr[..., :3]
	buf.close()
	h, w, c = img_arr.shape
	if h > w:
		padh = (h - w)//2
		padw = padh if (h - w) % 2 == 0 else (h - w)//2 + 1
		img_arr = np.pad(img_arr, ((0,0),(padh,padw),(0,0)), 'constant', constant_values=(255, 255))
	elif w > h:
		padh = (w - h)//2
		padw = padh if (w - h) % 2 == 0 else (w - h)//2 + 1
		img_arr = np.pad(img_arr, ((padh,padw),(0,0),(0,0)), 'constant', constant_values=(255, 255))
	img = PIL.Image.fromarray(img_arr)
	img = img.resize(output_shape, PIL.Image.BILINEAR)
	img_arr = np.array(img)
	return img_arr

class VideoWriter:
	def __init__(self, filename='results/output.mp4', fps=30.0, **kw):
		self.writer = None
		self.params = dict(filename=filename, fps=fps, **kw)

	def add(self, img):
		img = np.asarray(img)
		if self.writer is None:
			h, w = img.shape[:2]
			self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
		if img.dtype in [np.float32, np.float64]:
			img = np.uint8(img.clip(0, 1)*255)
		if len(img.shape) == 2:
			img = np.repeat(img[..., None], 3, -1)
		elif img.shape[-1] == 1:
			img = np.repeat(img, 3, -1)
		elif img.shape[-1] == 4:
			img = img[..., :3]
		self.writer.write_frame(img)

	def close(self):
		if self.writer:
			self.writer.close()

	def __enter__(self):
		return self

	def __exit__(self, *kw):
		self.close()

"""
Tensor-based utility functions
"""

def to_nchw(img):
	img = torch.as_tensor(img)
	if len(img.shape) == 3:
		img = img[None,...]
	return img.permute(0, 3, 1, 2)

def to_nhwc(img):
	img = torch.as_tensor(img)
	if len(img.shape) == 3:
		img = img[None,...]
	return img.permute(0, 2, 3, 1)

def tensor2vec(x):
	b = x.shape[0]
	return x.reshape(b, -1, 1)

def vec2tensor(x, shape):
	b = x.shape[0]
	return x.reshape(b, *shape)

def to_uint8(x):
	if x.dtype in [torch.float32, torch.float64]:
		x = torch.uint8(torch.clip(x, 0, 1)*255)
	return x

def mono_to_rgb(img):
	assert img.ndim == 4
	if img.shape[1] == 1:
		return img.tile(1, 3, 1, 1)
	return img

def norm_grad(net):
	for name, p in net.named_parameters():
		if p.grad is not None and p.requires_grad:
			p.grad /= (p.grad.norm() + 1e-8)

def draw_msg(img, msg, where='append', font=None):
	'''img is [c, h, w] tensor with value range [0, 1]'''
	if img.shape[0] == 1:
		img = img.tile([3, 1, 1])
	W, H = img.shape[2], 15
	text_img = PIL.Image.new('RGB', (W, H), 'white')
	draw = PIL.ImageDraw.Draw(text_img)
	w, h = font.getsize(msg) if font is not None else draw.textsize(msg)
	draw.text(((W-w)//2, (H-h)//2), msg, fill='black', font=font)
	text_img = to_tensor(text_img).to(img.device)
	if where == 'append':
		img = torch.cat([img, text_img], 1)
	elif where == 'prepend':
		img = torch.cat([text_img, img], 1)
	return img

def viz_attn_maps(attn_maps, attn_size, images, blend=False, alpha=0.85, brighten=1.):
	"""
	Args:
		attn_maps: 5-D float32 Tensor of shape [B, heads, img_height*img_width, 1, attn_height*attn_width]
		attn_size: [attn_height, attn_width]
		images: a batch of float32 Tensors of shape [B, 3 (or 1), image_height, image_width]
	Returns:
		colour_attn_maps: a float32 Tensor of shape [heads*B, 3, input_height, input_width]
	"""
	b, c, img_height, img_width = images.shape
	heads = attn_maps.shape[1]
	attn_padding = (attn_size[0]//2, attn_size[1]//2)
	device = attn_maps.device

	# b heads (img_h img_w) 1 (attn_height attn_width) -> b (heads attn_height attn_width) (img_h img_w)
	attn_maps = rearrange(attn_maps.squeeze(3), 'b h n a -> b (h a) n')

	# Locations where there's attention > 0
	locs = torch.where(
		attn_maps > 0.0,
		torch.tensor(1.0, dtype=attn_maps.dtype, device=device),
		torch.tensor(0.0, dtype=attn_maps.dtype, device=device)
	)

	# Divisor based on number of times a location has had attention added onto it
	# b (heads attn_height attn_width) (img_h img_w) -> b heads 1 img_h img_w
	divisor = fold(locs, output_size=(img_height, img_width), kernel_size=attn_size, padding=attn_padding)
	divisor = torch.where(
		divisor==0,
		torch.tensor(1.0, dtype=attn_maps.dtype, device=device),
		divisor
	).unsqueeze(2)  # converting zeros to ones

	# colourize and blend regions with overlapping attn
	rg = xy_meshgrid(img_width, img_height, 0, 1, 0, 1, b, device=device)
	b = torch.linspace(1, 0, img_width, dtype=rg.dtype, device=rg.device) \
		.reshape(1, 1, 1, img_width).tile(b, 1, img_height, 1)
	b = torch.split(b, 1, dim=1)  # List[(b 1 img_h img_w)]
	rgb_cubes = list(map(lambda _b: torch.cat([rg, _b], 1), b))  # List[(b 3 img_h img_w)]
	rgb_cubes = torch.stack(rgb_cubes, 1).unsqueeze(2)  # b heads 1 3 img_h img_w

	# b (heads attn_height attn_width) (img_h img_w) -> b heads (attn_height attn_width) 3 img_h img_w
	attn_maps = rearrange(
		attn_maps,
		'b (h a_h a_w) (i_h i_w) -> b h (a_h a_w) 1 i_h i_w',
		h=heads, a_h=attn_size[0], a_w=attn_size[1], i_h=img_height, i_w=img_width
	).tile(1, 1, 1, 3, 1, 1)
	attn_maps = attn_maps * rgb_cubes * brighten
	# attn_maps = torch.ones_like(attn_maps) * rgb_cubes

	# b heads (attn_height attn_width) 3 img_h img_w -> b (heads attn_height attn_width 3) (img_h img_w)
	attn_maps = rearrange(
		attn_maps,
		'b h (a_h a_w) c i_h i_w -> b (c h a_h a_w) (i_h i_w)',
		h=heads, a_h=attn_size[0], a_w=attn_size[1], i_h=img_height, i_w=img_width
	)

	# b (3 heads attn_height attn_width) (img_h img_w) -> b (3 heads) img_h img_w
	attn_maps = fold(attn_maps, output_size=(img_height, img_width), kernel_size=attn_size, padding=attn_padding)

	# b (3 heads) img_h img_w -> b heads 3 img_h img_w
	attn_maps = rearrange(attn_maps, 'b (c h) i_h i_w -> b h c i_h i_w', h=heads)
	attn_maps = attn_maps / divisor

	if blend:
		if c == 1:
			images = images.tile(1, 3, 1, 1)
		images = images.unsqueeze(1)  # b 1 3 img_h img_w
		blended = (alpha*attn_maps) + ((1-alpha)*images)  # b heads 3 img_h img_w
		blended = rearrange(blended.transpose(0, 1), 'h b c i_h i_w -> (h b) c i_h i_w')
		return blended
	else:
		attn_maps = rearrange(attn_maps.transpose(0, 1), 'h b c i_h i_w -> (h b) c i_h i_w')
		return attn_maps


"""
Convolutional preprocessing with hand-crafted filters
"""

def perception(x, filters):
	'''filters: [filter_n, h, w]'''
	if x.dtype == torch.float64:
		filters = filters.double()
	b, c, h, w = x.shape
	y = torch.nn.functional.conv2d(x.reshape(b*c, 1, h, w), filters[:, None], padding='same')
	return y.reshape(b, -1, h, w)

def setup_conv_preprocessing(device):
	ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]], device=device)
	sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]], device=device)/8.0
	lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]], device=device)/16.0
	filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
	_perception = lambda x : perception(x, filters)  # TODO: switch to functools.partial
	return _perception

"""
Helper functions for training
"""
# TODO: better documentation
def masking_schedule(i, schedule_start=500, schedule_end=10000, prob_stages=3, max_prob=0.75, patch_shape_stages=3,
					 max_patch_shape=(10, 10), random_seed=None):
	"""
	Masking schedule gets geometrically harder as i gets larger. The number of choices of masking strategies increases
	as i increases, and they get more challenging. A single one is picked from the pool at random.
	"""
	start_prob = 0.25
	start_patch_shape = (1, 1)
	if i == -1:
		i = schedule_end+1
	probs = np.linspace(start_prob, max_prob, num=prob_stages, dtype=np.float32)
	# TODO: get irregular patch shape combinations working (i.e., separate patch height and width)
	patch_shapes = np.linspace(start_patch_shape[0], max_patch_shape[0], num=patch_shape_stages,
							   dtype=np.float32).astype(np.int32)
	combs = list(itertools.product(probs, patch_shapes))
	sched = np.geomspace(schedule_start, schedule_end, num=len(combs), dtype=np.float32).astype(np.int32)
	for idx, stage in enumerate(sched):
		if i <= stage:
			combs = combs[:idx+1]
			break
	if random_seed is not None:
		random.seed(random_seed)
	p, p_s = random.choice(combs)
	return p, pair(p_s)

def checkpoint_sequential(function, x, segments, seq_length, **kwargs):
	# Hack for keyword-only parameter in a python 2.7-compliant way
	preserve = kwargs.pop('preserve_rng_state', True)

	def run_function(start, end, function, **kwargs):
		def forward(x):
			for _ in range(start, end + 1):
				x = function(x, **kwargs)
			return x
		return forward

	segment_size = seq_length // segments
	# the last chunk has to be non-volatile
	end = -1
	for start in range(0, segment_size * (segments - 1), segment_size):
		end = start + segment_size - 1
		x = torch.utils.checkpoint.checkpoint(run_function(start, end, function, **kwargs), x,
											  preserve_rng_state=preserve)
	return run_function(end + 1, seq_length - 1, function)(x)

"""
ViT NCA helper functions
"""
def xy_meshgrid(width, height, minval_x=-1., maxval_x=1.,
				minval_y=-1., maxval_y=1., batch_size=1, device='cpu'):
	x_coords, y_coords = \
		torch.meshgrid(torch.linspace(minval_x, maxval_x, width, device=device),
					   torch.linspace(minval_y, maxval_y, height, device=device), indexing='xy')
	xy_coords = torch.stack([x_coords, y_coords], 0)  # [2, height, width]
	xy_coords = torch.unsqueeze(xy_coords, 0) # [1, 2, height, width]
	xy_coords = torch.tile(xy_coords, [batch_size, 1, 1, 1])
	return xy_coords  # +x right, +y down, xy-indexed

def xyz_meshgrid_2(width, height, depth,
				   minval_x=-1., maxval_x=1.,
				   minval_y=-1., maxval_y=1.,
				   minval_z=-1., maxval_z=1.,
				   batch_size=1, device='cpu'):
	xy_coords = xy_meshgrid(width, height, minval_x, maxval_x, minval_y, maxval_y, batch_size, device)
	xy_coords = xy_coords.unsqueeze(1).tile(1, depth, 1, 1, 1)
	z_coords = torch.linspace(minval_z, maxval_z, depth, device=device).reshape(1, depth, 1, 1)\
					.tile(1, 1, height, width).unsqueeze(2)
	xyz_coords = torch.cat([xy_coords, z_coords], dim=2)
	return xyz_coords

def xyz_meshgrid(width, height, depth,
				 minval_x=-1., maxval_x=1.,
				 minval_y=-1., maxval_y=1.,
				 minval_z=-1., maxval_z=1.,
				 batch_size=1, device='cpu'):
	x_coords, y_coords, z_coords = torch.meshgrid(
		torch.linspace(minval_x, maxval_x, width, device=device),
		torch.linspace(minval_y, maxval_y, height, device=device),
		torch.linspace(minval_z, maxval_z, depth, device=device),
		indexing='xy'
	)
	xyz_coords = torch.stack([x_coords, y_coords, z_coords], 0)
	xyz_coords = rearrange(xyz_coords, 'c h w s -> s c h w')
	xyz_coords = repeat(xyz_coords, 's c h w -> b s c h w', b=batch_size)
	return xyz_coords

def nerf_positional_encoding(xy_coords, L=10, basis_function='sin_cos', device='cpu'):
	if basis_function == 'raw_xy':
		return xy_coords
	elif basis_function == 'sin_cos':
		l = torch.arange(L, device=device).reshape(1, L, 1, 1)
		x = 2**l * torch.pi * xy_coords[:, 0:1]
		y = 2**l * torch.pi * xy_coords[:, 1:2]
		xy = torch.cat([x, y], 1)
		pe = torch.cat([torch.sin(xy), torch.cos(xy)], 1)
	elif basis_function == 'sinc':
		l = torch.arange(L, device=device).reshape(1, L, 1, 1)
		x = 2**l * torch.pi * xy_coords[:, 0:1]
		y = 2**l * torch.pi * xy_coords[:, 1:2]
		xy = torch.cat([x, y], 1)
		pe = torch.cat([torch.special.sinc(xy), torch.special.sinc(xy + torch.pi/2.0)], 1)
	elif basis_function == 'sin_cos_xy':
		l = torch.arange(L, device=device).reshape(1, L, 1, 1)
		x = 2**l * torch.pi * xy_coords[:, 0:1]
		y = 2**l * torch.pi * xy_coords[:, 1:2]
		xy = torch.cat([x, y], 1)
		pe = torch.cat([torch.sin(xy), torch.cos(xy), xy_coords], 1)
	return pe

def vit_positional_encoding(n, dim, device='cpu'):
	position = torch.arange(n, device=device).unsqueeze(1)
	div_term_even = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
	div_term_odd = torch.exp(torch.arange(1, dim, 2, device=device) * (-math.log(10000.0) / dim))
	pe = torch.zeros(n, 1, dim, device=device)
	pe[:, 0, 0::2] = torch.sin(position * div_term_even)
	pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
	return pe.transpose(0, 1)

def neighbourhood_filters(neighbourhood_size, device):
	height, width = neighbourhood_size
	impulses = []
	for i in range(height):
		for j in range(width):
			impulse = signal.unit_impulse((height, width), idx=(i,j), dtype=np.float32)
			impulses.append(impulse)
	filters = torch.tensor(np.stack(impulses), device=device)
	return filters

class ExtractOverlappingPatches(torch.nn.Module):
	def __init__(self, patch_size, device) -> None:
		super().__init__()
		self.patch_size = patch_size
		self.device = device
		self.filters = neighbourhood_filters(self.patch_size, self.device)

	def forward(self, x):
		'''filters: [filter_n, h, w]'''
		b = x.shape[0]
		y = rearrange(x, 'b c h w -> (b c) 1 h w')
		y = torch.nn.functional.conv2d(y, self.filters[:, None], padding='same')
		_x = rearrange(y, '(b c) filter_n h w -> b (c filter_n) h w', b=b)
		return _x

class LocalizeAttention(torch.nn.Module):
	def __init__(self, attn_neighbourhood_size, device) -> None:
		super().__init__()
		self.attn_neighbourhood_size = attn_neighbourhood_size
		self.device = device
		self.attn_filters = neighbourhood_filters(self.attn_neighbourhood_size, self.device)

	def forward(self, x, height, width):
		'''attn_filters: [filter_n, h, w]'''
		b, h, _, d = x.shape
		y = rearrange(x, 'b h (i j) d -> (b h d) 1 i j', i=height, j=width)
		y = torch.nn.functional.conv2d(y, self.attn_filters[:, None], padding='same')
		_x = rearrange(y, '(b h d) filter_n i j -> b h (i j) filter_n d', b=b, h=h, d=d)
		return _x

# A little bit slower than above implementation
def localize_attention(x, input_size, attn_size):
	b, h, _, d = x.shape
	y = rearrange(x, 'b h (i j) d -> b (h d) i j', i=input_size[0], j=input_size[1])

	# (b h d) attn_size input_size
	y = unfold(y, kernel_size=attn_size, padding=(attn_size[0]//2, attn_size[1]//2))

	_x = rearrange(y, 'b (h d attn_size) n -> b h n attn_size d', b=b, h=h, d=d)
	return _x