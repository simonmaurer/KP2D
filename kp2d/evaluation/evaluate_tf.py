#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import torch
from tqdm import tqdm

from kp2d.evaluation.descriptor_evaluation_tf import (compute_homography, compute_matching_score)
from kp2d.evaluation.detector_evaluation_tf import compute_repeatability
#from kp2d.utils.image import to_color_normalized, to_gray_normalized


def evaluate_model(dataset, model, output_shape=(320, 240), top_k=300, use_color=True):
    """Model evaluation script.
    
    Parameters
    ----------
    dataset: tf.data.datasets.Dataset
        Dataset. yielding images with `shape` (B, H, W, C), transformed_images with `shape (B, 5, H, W, C), gt_homographies with `shape` (B, 5, 3, 3)
    model: tf.keras.models.Model
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.    
    use_color: bool
        Use color or grayscale images.
    """

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

    for i, (ref_images, transformed_images, homographies) in tqdm(enumerate(dataset), desc="evaluate_model"):
    #for i, (ref_images, transformed_images, homographies) in enumerate(dataset):
        # sample
        for image_ref, images_transformed, homographies_transformed in zip(ref_images, transformed_images, homographies):
            image_ref = tf.expand_dims(image_ref, axis=0)
            scores_ref, positions_ref, descriptors_ref = model(image_ref, training=False)
            
            # Scores & Descriptors
            scores_shape = tf.shape(scores_ref)
            positions_shape = tf.shape(positions_ref)
            if len(scores_shape) == 4:  # if (B, H, W, 1) and not (B, num_keypoints, 1)
                scores_ref = tf.reshape(scores_ref, (scores_shape[0], scores_shape[1]*scores_shape[2], scores_shape[3]))
                positions_ref = tf.reshape(positions_ref, (positions_shape[0], positions_shape[1]*positions_shape[2], positions_shape[3]))
            scores_ref = tf.concat([positions_ref, scores_ref], axis=-1)
            scores_ref = tf.squeeze(scores_ref).numpy()
            
            descriptors_shape = tf.shape(descriptors_ref)
            if len(descriptors_shape) == 4:  # if (B, H, W, C) and not (B, num_keypoints, C)
                descriptors_ref = tf.reshape(descriptors_ref, (descriptors_shape[0], descriptors_shape[1]*descriptors_shape[2], descriptors_shape[3]))
            descriptors_ref = tf.squeeze(descriptors_ref).numpy()
            
            # Filter based on confidence threshold
            descriptors_ref = descriptors_ref[scores_ref[:, 2] > conf_threshold, :]
            scores_ref = scores_ref[scores_ref[:, 2] > conf_threshold, :]
            #B, C, Hc, Wc = descriptors_ref.shape
            
            ##
            #if use_color:
            #    image = to_color_normalized(sample['image'].cuda())
            #    warped_image = to_color_normalized(sample['warped_image'].cuda())
            #else:
            #    image = to_gray_normalized(sample['image'].cuda())
            #    warped_image = to_gray_normalized(sample['warped_image'].cuda())
            ##
            
            for image_transformed, H in zip(images_transformed, homographies_transformed):
                image_transformed = tf.expand_dims(image_transformed, axis=0)
                scores_transformed, positions_transformed, descriptors_transformed = model(image_transformed, training=False)
            
                # Scores & Descriptors
                scores_transformed_shape = tf.shape(scores_transformed)
                positions_transformed_shape = tf.shape(positions_transformed)
                if len(scores_transformed_shape) == 4:  # if (B, H, W, 1) and not (B, num_keypoints, 1)
                    scores_transformed = tf.reshape(scores_transformed, (scores_transformed_shape[0], scores_transformed_shape[1]*scores_transformed_shape[2], scores_transformed_shape[3]))
                    positions_transformed = tf.reshape(positions_transformed, (positions_transformed_shape[0], positions_transformed_shape[1]*positions_transformed_shape[2], positions_transformed_shape[3]))
                scores_transformed = tf.concat([positions_transformed, scores_transformed], axis=-1)
                scores_transformed = tf.squeeze(scores_transformed).numpy()
                #print('Trafo: ')
                #print(scores_transformed.shape)
            
                descriptors_transformed_shape = tf.shape(descriptors_transformed)
                if len(descriptors_transformed_shape) == 4:  # if (B, H, W, C) and not (B, num_keypoints, C)
                    descriptors_transformed = tf.reshape(descriptors_transformed, (descriptors_transformed_shape[0], descriptors_transformed_shape[1]*descriptors_transformed_shape[2], descriptors_transformed_shape[3]))
                descriptors_transformed = tf.squeeze(descriptors_transformed).numpy()
            
                # Filter based on confidence threshold
                descriptors_transformed = descriptors_transformed[scores_transformed[:, 2] > conf_threshold, :]
                scores_transformed = scores_transformed[scores_transformed[:, 2] > conf_threshold, :]
                
                # Prepare data for evaluation
                data = {'image': image_ref.numpy().squeeze(),
                        'image_shape' : output_shape,
                        'warped_image': image_transformed.numpy().squeeze(),
                        'homography': H.numpy(),
                        'prob': scores_ref, 
                        'warped_prob': scores_transformed,
                        'desc': descriptors_ref,
                        'warped_desc': descriptors_transformed}
            
                # Compute repeatabilty and localization error
                _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
                repeatability.append(rep)
                localization_err.append(loc_err)
                #print('Repeatability: ')
                #print(np.mean(repeatability))
                #print('Loc error: ')
                #print(np.mean(localization_err))

                # Compute correctness
                c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
                correctness1.append(c1)
                correctness3.append(c2)
                correctness5.append(c3)

                # Compute matching score
                mscore = compute_matching_score(data, keep_k_points=top_k)
                MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)

def evaluate_model_alternative(dataset, model, output_shape=(320, 240), top_k=300, use_color=True):
    """Model evaluation script.
    
    Parameters
    ----------
    dataset: tf.data.datasets.Dataset
        Dataset. yielding images with `shape` (B, H, W, C), transformed_images with `shape (B, H, W, C), gt_homographies with `shape` (B, 3, 3)
    model: tf.keras.models.Model
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.    
    use_color: bool
        Use color or grayscale images.
    """

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

    for i, (ref_images, transformed_images, homographies) in tqdm(enumerate(dataset), desc="evaluate_model_alternative"):
    #for i, (ref_images, transformed_images, homographies) in enumerate(dataset):
        # sample
        for image_ref, images_transformed, homographies_transformed in zip(ref_images, transformed_images, homographies):
            image_ref = tf.expand_dims(image_ref, axis=0)
            image_transformed = tf.expand_dims(image_transformed, axis=0)
            scores_ref, positions_ref, descriptors_ref = model(image_ref, training=False)
            scores_transformed, positions_transformed, descriptors_transformed = model(image_transformed, training=False)
            
            # Scores & Descriptors
            scores_shape = tf.shape(scores_ref)
            positions_shape = tf.shape(positions_ref)
            if len(scores_shape) == 4:  # if (B, H, W, 1) and not (B, num_keypoints, 1)
                scores_ref = tf.reshape(scores_ref, (scores_shape[0], scores_shape[1]*scores_shape[2], scores_shape[3]))
                positions_ref = tf.reshape(positions_ref, (positions_shape[0], positions_shape[1]*positions_shape[2], positions_shape[3]))
            scores_ref = tf.concat([positions_ref, scores_ref], axis=-1)
            scores_ref = tf.squeeze(scores_ref).numpy()
            
            scores_transformed_shape = tf.shape(scores_transformed)
            positions_transformed_shape = tf.shape(positions_transformed)
            if len(scores_transformed_shape) == 4:  # if (B, H, W, 1) and not (B, num_keypoints, 1)
                scores_transformed = tf.reshape(scores_transformed, (scores_transformed_shape[0], scores_transformed_shape[1]*scores_transformed_shape[2], scores_transformed_shape[3]))
                positions_transformed = tf.reshape(positions_transformed, (positions_transformed_shape[0], positions_transformed_shape[1]*positions_transformed_shape[2], positions_transformed_shape[3]))
            scores_transformed = tf.concat([positions_transformed, scores_transformed], axis=-1)
            scores_transformed = tf.squeeze(scores_transformed).numpy()
            
            descriptors_shape = tf.shape(descriptors_ref)
            if len(descriptors_shape) == 4:  # if (B, H, W, C) and not (B, num_keypoints, C)
                descriptors_ref = tf.reshape(descriptors_ref, (descriptors_shape[0], descriptors_shape[1]*descriptors_shape[2], descriptors_shape[3]))
            descriptors_ref = tf.squeeze(descriptors_ref).numpy()
            
            descriptors_transformed_shape = tf.shape(descriptors_transformed)
            if len(descriptors_transformed_shape) == 4:  # if (B, H, W, C) and not (B, num_keypoints, C)
                descriptors_transformed = tf.reshape(descriptors_transformed, (descriptors_transformed_shape[0], descriptors_transformed_shape[1]*descriptors_transformed_shape[2], descriptors_transformed_shape[3]))
            descriptors_transformed = tf.squeeze(descriptors_transformed).numpy()
            
            # Filter based on confidence threshold
            descriptors_ref = descriptors_ref[scores_ref[:, 2] > conf_threshold, :]
            scores_ref = scores_ref[scores_ref[:, 2] > conf_threshold, :]
            descriptors_transformed = descriptors_transformed[scores_transformed[:, 2] > conf_threshold, :]
            scores_transformed = scores_transformed[scores_transformed[:, 2] > conf_threshold, :]
            #B, C, Hc, Wc = descriptors_ref.shape
            
            # Prepare data for evaluation
            data = {'image': image_ref.numpy().squeeze(),
                    'image_shape' : output_shape,
                    'warped_image': image_transformed.numpy().squeeze(),
                    'homography': H.numpy(),
                    'prob': scores_ref, 
                    'warped_prob': scores_transformed,
                    'desc': descriptors_ref,
                    'warped_desc': descriptors_transformed}
            
            # Compute repeatabilty and localization error
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)
            
            ##
            #if use_color:
            #    image = to_color_normalized(sample['image'].cuda())
            #    warped_image = to_color_normalized(sample['warped_image'].cuda())
            #else:
            #    image = to_gray_normalized(sample['image'].cuda())
            #    warped_image = to_gray_normalized(sample['warped_image'].cuda())
            ##
            

    return np.mean(repeatability), np.mean(localization_err), np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)

def evaluate_step(images_ref, images_transformed, homographies, scores_ref_batch, positions_ref_batch, descriptors_ref_batch, scores_transformed_batch, positions_transformed_batch, descriptors_transformed_batch, output_shape=(320, 240), top_k=300):
    """Model evaluation script.
    
    Parameters
    ----------
    dataset: tf.data.datasets.Dataset
        Dataset. yielding images with `shape` (B, H, W, C), transformed_images with `shape (B, H, W, C), gt_homographies with `shape` (B, 3, 3)
    model: tf.keras.models.Model
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    """

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []
    
    B, _, _, _ = scores_ref_batch.shape
    
    for i in range(B):
        image_ref = images_ref[i]
        image_transformed = images_transformed[i]
        
        scores_ref, positions_ref, descriptors_ref = scores_ref_batch[i], positions_ref_batch[i], descriptors_ref_batch[i]
        scores_transformed, positions_transformed, descriptors_transformed = scores_transformed_batch[i], positions_transformed_batch[i], descriptors_transformed_batch[i]
        
        H = homographies[i]
        
        
        # Scores & Descriptors
        scores_shape = tf.shape(scores_ref)
        positions_shape = tf.shape(positions_ref)
        if len(scores_shape) == 3:  # if (H, W, 1) and not (num_keypoints, 1)
            scores_ref = tf.reshape(scores_ref, (scores_shape[0]*scores_shape[1], scores_shape[2]))
            positions_ref = tf.reshape(positions_ref, (positions_shape[0]*positions_shape[1], positions_shape[2]))
        scores_ref = tf.concat([positions_ref, scores_ref], axis=-1)
        scores_ref = scores_ref.numpy()
            
        scores_transformed_shape = tf.shape(scores_transformed)
        positions_transformed_shape = tf.shape(positions_transformed)
        if len(scores_transformed_shape) == 3:  # if (H, W, 1) and not (num_keypoints, 1)
            scores_transformed = tf.reshape(scores_transformed, (scores_transformed_shape[0]*scores_transformed_shape[1], scores_transformed_shape[2]))
            positions_transformed = tf.reshape(positions_transformed, (positions_transformed_shape[0]*positions_transformed_shape[1], positions_transformed_shape[2]))
        scores_transformed = tf.concat([positions_transformed, scores_transformed], axis=-1)
        scores_transformed = scores_transformed.numpy()
            
        descriptors_shape = tf.shape(descriptors_ref)
        if len(descriptors_shape) == 3:  # if (H, W, C) and not (num_keypoints, C)
            descriptors_ref = tf.reshape(descriptors_ref, (descriptors_shape[0]*descriptors_shape[1], descriptors_shape[2]))
        descriptors_ref = descriptors_ref.numpy()
            
        descriptors_transformed_shape = tf.shape(descriptors_transformed)
        if len(descriptors_transformed_shape) == 3:  # if (H, W, C) and not (num_keypoints, C)
            descriptors_transformed = tf.reshape(descriptors_transformed, (descriptors_transformed_shape[0]*descriptors_transformed_shape[1], descriptors_transformed_shape[2]))
        descriptors_transformed = descriptors_transformed.numpy()
            
        # Filter based on confidence threshold
        descriptors_ref = descriptors_ref[scores_ref[:, 2] > conf_threshold, :]
        scores_ref = scores_ref[scores_ref[:, 2] > conf_threshold, :]
        descriptors_transformed = descriptors_transformed[scores_transformed[:, 2] > conf_threshold, :]
        scores_transformed = scores_transformed[scores_transformed[:, 2] > conf_threshold, :]
        #B, C, Hc, Wc = descriptors_ref.shape
            
        # Prepare data for evaluation
        data = {'image': image_ref.numpy(),
                'image_shape' : output_shape,
                'warped_image': image_transformed.numpy(),
                'homography': H.numpy(),
                'prob': scores_ref, 
                'warped_prob': scores_transformed,
                'desc': descriptors_ref,
                'warped_desc': descriptors_transformed}
            
        # Compute repeatabilty and localization error
        _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
        repeatability.append(rep)
        localization_err.append(loc_err)

        # Compute correctness
        c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        # Compute matching score
        mscore = compute_matching_score(data, keep_k_points=top_k)
        MScore.append(mscore)
            

    return np.mean(repeatability), np.mean(localization_err), np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)
