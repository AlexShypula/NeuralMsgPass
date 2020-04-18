import os

PATH_TO_DATA = "mtl-dataset"
DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video', ]


for dataset_name in DATASETS:
    with open(os.path.join(PATH_TO_DATA, dataset_name) + '.task.train', \
        encoding='utf-8', errors='ignore') as f, \
        open(os.path.join(PATH_TO_DATA, dataset_name) + '.train', 'w') as t,\
        open(os.path.join(PATH_TO_DATA, dataset_name) + '.val', 'w') as v:
        length = len(f.readlines())
        f.seek(0)
        for i, line in enumerate(f):
            if i >= length - 200:
                v.write(line)
            else:
                t.write(line)
