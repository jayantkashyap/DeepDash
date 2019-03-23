import os
import cv2
from tqdm import tqdm

dataset_dir = 'repo'
entity_name = 'FlickrLogos-32'
data_type = 'train'

os.makedirs(os.path.join(dataset_dir, entity_name+'_', data_type))

data_extensions = ['jpg', 'png']

for item in list(os.walk(os.path.join(dataset_dir, entity_name)))[2:]:

    if (item[0].split(os.path.sep)[2] == 'val') or (item[0].split(os.path.sep)[2] == 'test'):
        continue

    if len(item[2]) > 0:
        label = item[0].split(os.path.sep)[-1]
        os.makedirs(os.path.join(dataset_dir, entity_name+'_', data_type, label))
        i=0
        for image_name in tqdm(item[2]):

        	if image_name.split('.')[-1] in data_extensions:
        		image = cv2.imread(os.path.join(item[0], image_name))
        		image = cv2.resize(image, (300, 300),interpolation=cv2.INTER_AREA)
        		# if i%4 == 0:
        		cv2.imwrite(f'{dataset_dir}//{entity_name}_//{data_type}//{label}//{i}.jpg', image)
        		i+=1
