import numpy as np
import random
import cv2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def shuffle_image_rows(img):
    row_permutation = np.arange(img.shape[0])
    np.random.shuffle(row_permutation)

    return img[row_permutation]

def lower_coverage(img):
    IMAGE_HEIGHT = len(img)
    LOWER_LIMIT_DIVIDER = 5
    UPPER_LIMIT_DIVIDER = 0.8

    lower_limit = 8
    upper_limit = int(round(IMAGE_HEIGHT * UPPER_LIMIT_DIVIDER))

    coverage_cutoff = random.randint(lower_limit, upper_limit)
    img[coverage_cutoff:,:] = 0

    return img

def find_lower_limit(img):
    for i, row in enumerate(img):
        if i <= 3:
            continue
        if np.sum(row) == 0:
            return i
            break
    
    return len(img)

def resize_height(img_limit, img):
    IMAGE_HEIGHT = len(img)
    IMAGE_WIDTH = len(img[0])

    cutoff_img = img[:img_limit,:]
    resized_img = cv2.resize(cutoff_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return resized_img


def match_heights(child, father, mother):
    IMAGE_HEIGHT = len(child) # Can use any image
    
    child_limit = find_lower_limit(child)
    father_limit = find_lower_limit(father)
    mother_limit = find_lower_limit(mother)

    if child_limit != IMAGE_HEIGHT:
        child = resize_height(child_limit, child)

    if father_limit != IMAGE_HEIGHT:
        father = resize_height(father_limit, father)

    if mother_limit != IMAGE_HEIGHT:
        mother = resize_height(mother_limit, mother)

    return child, father, mother

def mix_channels(child, father, mother, control):
    if control:
        channel_configuration = random.randint(0,2)

        if channel_configuration == 0:
            specific_configuration = random.randint(0,5)

            if specific_configuration == 0:
                child, father, mother = father, child, mother
            elif specific_configuration == 1:
                child, father, mother = mother, child, father
            elif specific_configuration == 2:
                child, father, mother = father, mother, child
            elif specific_configuration == 3:
                child, father, mother = mother, father, child
            elif specific_configuration == 4:
                child, father, mother = father, child, father
            elif specific_configuration == 5:
                child, father, mother = mother, child, mother

        elif channel_configuration == 1:
            specific_configuration = random.randint(0,5)

            if specific_configuration == 0:
                child, father, mother = child, child, mother
            elif specific_configuration == 1:
                child, father, mother = child, child, father
            elif specific_configuration == 2:
                child, father, mother = child, mother, child
            elif specific_configuration == 3:
                child, father, mother = child, father, child
            elif specific_configuration == 4:
                child, father, mother = mother, child, child
            elif specific_configuration == 5:
                child, father, mother = father, child, child

        elif channel_configuration == 2:
            child, father, mother = child, child, child

    elif not control:
        channel_configuration = random.randint(0,2)
        
        if channel_configuration == 0:
            child, father, mother = child, mother, father
        elif channel_configuration == 1:
            child, father, mother = child, father, father
        elif channel_configuration == 2:
            child, father, mother = child, mother, mother

    return child, father, mother

def reorder_channel(img, channel_permutation):

    new_img = np.zeros(img.shape)
    new_img[:,0::4], new_img[:,1::4], new_img[:,2::4], new_img[:,3::4] = img[:,channel_permutation[0]::4], img[:,channel_permutation[1]::4], img[:,channel_permutation[2]::4], img[:,channel_permutation[3]::4]

    return new_img

def combine_and_order_image_channels(child, father, mother, permutation=False):
    IMAGE_HEIGHT = len(child)
    IMAGE_WIDTH = len(child[0])
    IMAGE_CHANNELS = 3

    full_img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    channel_permutation = np.array([0,1,2,3])

    if permutation:
        channel_permutation = np.random.permutation(channel_permutation)

    # Match heights
    # child, father, mother = match_heights(child, father, mother)

    full_img[:,:,0] = child
    full_img[:,:,1] = father
    full_img[:,:,2] = mother

    full_img[:,:,0] = reorder_channel(full_img[:,:,0], channel_permutation)
    full_img[:,:,1] = reorder_channel(full_img[:,:,1], channel_permutation)
    full_img[:,:,2] = reorder_channel(full_img[:,:,2], channel_permutation)
    
    full_img = reorder_channel(full_img, channel_permutation)
    
    return full_img

def horizontal_flip(img):
    img = np.flip(img, axis=1)
    return img

def image_to_ascii(img):
    central_position = int(len(img[0]) / 4) // 2
    
    for row in img:
        temp_pileup = ''
        for position_index, position_value  in enumerate(row[::4]):
            if row[position_index * 4 + 0] == 1:
                base = 'A'
            elif row[position_index * 4 + 1] == 1:
                base = 'C'
            elif row[position_index * 4 + 2] == 1:
                base = 'T'
            elif row[position_index * 4 + 3] == 1:
                base = 'G'
            else:
                base = '   '

            if position_index == central_position:
                temp_pileup += bcolors.WARNING + base + '  ' + bcolors.ENDC
            else:
                temp_pileup += base + '  '
        
        print(temp_pileup)


def decode_channels(full_img):

    child, mother, father = full_img[:,:,0], full_img[:,:,1], full_img[:,:,2]

    return child, father, mother

def augment_img(child, father, mother, control, reporting=False):

    # Decide coverage
    low_coverage_coinflip = random.randint(0,3)
    if low_coverage_coinflip == 0:
        low_coverage = True
    else:
        low_coverage = False

    # Decide flip
    # flip_coinflip = random.randint(0,1)
    # if flip_coinflip == 1:
    #     flip = True
    # else:
    #     flip = False
    

    # 1. Mix channels
    # child, father, mother = mix_channels(child, father, mother, control)

    # 3. Coverage
    if low_coverage:
        child, father, mother = lower_coverage(child), lower_coverage(father), lower_coverage(mother)

    # 4. Permutation
    full_img = combine_and_order_image_channels(child, father, mother, permutation=True)

    # 5. Flip
    # if flip:
    #     full_img = horizontal_flip(full_img)

    if reporting:
        print('Low coverage:',low_coverage)
        print('Flip:',flip)
        # child, father, mother = decode_channels(full_img)
        
        print('child')
        image_to_ascii(child)
        print('father')
        image_to_ascii(father)
        print('mother')
        image_to_ascii(mother)
        print()

    return full_img
