from lib import *


train_transform = T.Compose([

    T.Resize(size=(CFG.img_size, CFG.img_size)),  # Resizing the image to be 224 by 224
    T.RandomRotation(degrees=(-20, +20)),
    # Randomly Rotate Images by +/- 20 degrees, Image argumentation for each epoch
    T.ToTensor(),
    # converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Normalize by 3 means 3 StD's of the image net, 3 channels

])

validate_transform = T.Compose([

    T.Resize(size=(CFG.img_size, CFG.img_size)),  # Resizing the image to be 224 by 224
    # T.RandomRotation(degrees=(-20,+20)), #NO need for validation
    T.ToTensor(),
    # converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Normalize by 3 means 3 StD's of the image net, 3 channels

])

test_transform = T.Compose([

    T.Resize(size=(CFG.img_size, CFG.img_size)),  # Resizing the image to be 224 by 224
    # T.RandomRotation(degrees=(-20,+20)), #NO need for validation
    T.ToTensor(),
    # converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Normalize by 3 means 3 StD's of the image net, 3 channels

])