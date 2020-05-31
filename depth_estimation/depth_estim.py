# THIS IS A CUSTOM AUGR FILE TO STREAMLINE USE OF THIS MODEL

# import depth_estimation.networks
import depth_estimation.networks as networks
import numpy as np
import torch
from torchvision import transforms
import cv2
import PIL.Image as pil
from sklearn.cluster import KMeans
from pandas import DataFrame
from matplotlib import pyplot as plt

def load_model(model_folder='depth_estimation/models/mono+stereo_640x192'):
    """
        Load a model from a given folder path
    """

    encoder_path = '{}/{}'.format(model_folder, 'encoder.pth')
    decoder_path = '{}/{}'.format(model_folder, 'depth.pth')

    encoder = networks.resnet_encoder.ResnetEncoder(18, False)
    depth_decoder = networks.depth_decoder.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))


    # TODO support CUDA
    enc_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    filtered_dict_enc = {k: v for k, v in enc_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    dep_dict = torch.load(decoder_path, map_location=torch.device('cpu'))
    depth_decoder.load_state_dict(dep_dict)

    encoder.eval()
    depth_decoder.eval()

    return encoder,depth_decoder

def get_depth_frame(encoder, decoder, frame, target_shape=(640,192)):
    """
        Given a depth encoder, decoder, and a frame, return a
        numpy array of equal shape to `frame` that contains the relative
        depth estimations
    """

    # TODO make this dynamic
    min_depth = 0.5
    max_depth = 3

    frame = pil.fromarray(frame).convert('RGB')
    original_width, original_height = frame.size
    

    feed_width,feed_height = target_shape
    input_image_resized = frame.resize((feed_width, feed_height), pil.LANCZOS)
    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = decoder(features)

    disp = outputs[("disp", 0)]

    # depth_arr = (depth.numpy()[0,:,:,:] * 255).astype(np.uint8)
    depth_arr = disp.squeeze().numpy()
    # depth_arr = np.transpose(depth_arr, (1, 2, 0))

    resized_depth = cv2.resize(depth_arr, (original_width,original_height))

    scaled = disp_to_depth(resized_depth, min_depth, max_depth)[1]
    # cv2.imshow("scaled", resized_depth)
    return scaled

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def get_detection_depth(frame, person):
    detection_frame = person._get_subset_from_frame(frame, False)
    cluster_1,cluster_2 = separate_background_foreground(detection_frame)
    detection_depth = min(cluster_1, cluster_2)
    print(detection_depth)
    return detection_depth

def separate_background_foreground(frame):
    cluster_centers_ = get_clusters(frame)
    background = np.mean(cluster_centers_[0])
    foreground = np.mean(cluster_centers_[1])    
    return background, foreground

def get_clusters(pts):
    df = DataFrame(pts)
    kmeans = KMeans(n_clusters=2, max_iter=2000000).fit(df)
    return kmeans.cluster_centers_