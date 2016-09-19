'''
Author: Sanket Satpathy
Last updated: 6/8/16
Used with Python 2.7
Description: 
This is a script to generate posters for the ELE face detection project.
Posters are generated for each training image.
'''
import numpy as np, cv2, os
np.set_printoptions(precision=2)
from PIL import Image
from recognizer_util import filename_to_name

# make poster for display
def make_poster(image_dir, filename):
	height = 1080
	width = 1920
	shape = (height, width, 3)
	y = cv2.imread('./poster/nnet.tiff')
	y_shape = (791, 388)
	y = cv2.resize(y, y_shape)

	x_offset = 175
	y_offset = (width-y.shape[1])/2

	screen = 255 * np.ones(shape, dtype='uint8')

	#title
	font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
	cv2.putText(screen, 'ELE Face Detection', (height/2, 100), font, 3, (0,0,0), 3)

	# input image
	z = Image.open(image_dir + filename)
	aspect_ratio = z.size[1]/float(z.size[0])
	z = z.resize((200, int(200*aspect_ratio)))
	z = np.array(z, dtype='uint8')[:,:,::-1]
	screen[(x_offset+y.shape[0]/2)-z.shape[0]/2:(x_offset+y.shape[0]/2)-z.shape[0]/2+z.shape[0], y_offset-2*z.shape[1]:y_offset-z.shape[1]] = z

	# neural net
	# screen[x_offset:x_offset+y.shape[0], y_offset:y_offset+y.shape[1]] = y
	screen[x_offset:x_offset+y.shape[0], y_offset-30:y_offset-30+y.shape[1]] = y

	# prediction text
	font = cv2.FONT_HERSHEY_SIMPLEX
	if len(filename[:-4].split('_')) > 3:
		first = ' '.join(filename[:-4].split('_')[1:3])
		last = ' '.join(filename[:-4].split('_')[3:])
	else:
		first = filename[:-4].split('_')[1]
		last = ' '.join(filename[:-4].split('_')[2:])

	cv2.putText(screen, first, (y_offset + 9*y.shape[1]/8 - 50, x_offset + y.shape[0]/2 - 50), font, 1.5, (255,0,0), 3)
	cv2.putText(screen, last, (y_offset + 9*y.shape[1]/8 - 50, x_offset + y.shape[0]/2 + 20), font, 1.5, (255,0,0), 3)

	# explanation
	font = cv2.FONT_HERSHEY_TRIPLEX
	font_size = 1.2
	text_offset = 400
	x_offset += 25#50
	cv2.putText(screen, 'Transfer learning using pre-trained neural networks:', (text_offset, x_offset + y.shape[0] + 30), font, font_size, (0,0,0), 2)
	cv2.putText(screen, '  ~ OpenFace from Carnegie Mellon University', (text_offset, x_offset + y.shape[0] + 100), font, font_size, (0,0,0), 2)
	cv2.putText(screen, '  ~ VGG-Face from the University of Oxford', (text_offset, x_offset + y.shape[0] + 170), font, font_size, (0,0,0), 2)

	x_offset += 25

	# credits
	font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
	z = Image.open('./poster/kevin.png')
	aspect_ratio = float(z.size[0])/z.size[1]
	z = z.resize((int(200*aspect_ratio), 200))
	z = np.array(z, dtype='uint8')[:,:,::-1]
	image_xoffset = width/2 - int(2.5*z.shape[1])
	image_yoffset = 350
	font_size = 1
	
	cv2.line(screen, (100, (x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15-40), (screen.shape[1]-100,(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15-40), (0,0,0))
	cv2.line(screen, (100, (x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15-42), (screen.shape[1]-100,(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15-42), (0,0,0))
	cv2.putText(screen, 'Crafted by', (image_xoffset-400, (x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15+100), font, font_size+1, (0,0,0), 2)
	
	cv2.putText(screen, 'Kevin Wang', (image_xoffset, (x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15), font, font_size, (0,0,0), 2)
	screen[(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2:(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2+z.shape[0], image_xoffset:image_xoffset+z.shape[1]] = z

	image_xoffset += 2*z.shape[1]

	z = Image.open('./poster/sai.jpg')
	aspect_ratio = float(z.size[0])/z.size[1]
	z = z.resize((int(200*aspect_ratio), 200))
	z = np.array(z, dtype='uint8')[:,:,::-1]
	cv2.putText(screen, 'Sai Satpathy', (image_xoffset, (x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15), font, font_size, (0,0,0), 2)
	screen[(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2:(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2+z.shape[0], image_xoffset:image_xoffset+z.shape[1]] = z

	image_xoffset += 2*z.shape[1]

	z = Image.open('./poster/paul.png')
	aspect_ratio = float(z.size[0])/z.size[1]
	z = z.resize((int(200*aspect_ratio), 200))
	z = np.array(z, dtype='uint8')[:,:,::-1]
	if z.shape[2] == 4:
		z = z[:,:,1:]

	cv2.putText(screen, 'Paul Cuff', (image_xoffset, (x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2-15), font, font_size, (0,0,0), 2)
	screen[(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2:(x_offset+y.shape[0]+image_yoffset)-z.shape[0]/2+z.shape[0], image_xoffset:image_xoffset+z.shape[1]] = z

	cv2.imwrite('/Users/princetonee/Dropbox/EEdisplayfaces/poster/{0:s}.jpg'.format('poster_'+filename_to_name(filename)), screen)

def main():
	image_dir = '/Users/princetonee/Dropbox/EEdisplayfaces/'
	for filename in os.listdir(image_dir):
		if filename[-4:].lower() in ['.jpg', '.bmp', '.png', '.gif', '.ppm']:
			make_poster(image_dir, filename)

if __name__ == '__main__':
	main()
