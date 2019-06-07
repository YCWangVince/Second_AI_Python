# crop video of any size to 64 x 64 required for input to the model
import cv2
def six_four_crop_video(newVideo):
    """
    Args:
        newVideo --> type: moviepy VideoFileClip
    Funtionality:
        pass in video of any size. Result is a video transformed and resized to the ideal
        64 x 64 resolution required for input to the convolutional neural network
    """
    dimensions = newVideo.shape

    smaller_dimension = min(dimensions[0],dimensions[1])
    target = 64
    factor = smaller_dimension/target
    factored_dimensions = []
    for i in range(2):
        factored_dimensions.append(round(dimensions[i]/factor))
    first_stage_crop = cv2.resize(newVideo, (factored_dimensions[0], factored_dimensions[1]),interpolation=cv2.INTER_CUBIC)
    larger_dimension = max(factored_dimensions)
    midpoint = round(larger_dimension/2)
    limit = target/2
    lower = midpoint - limit #x1
    upper = midpoint + limit #x2
    if first_stage_crop.shape[0]>first_stage_crop.shape[1]:
        six_four_crop = first_stage_crop[int(lower):int(upper), 0:target, :]
    else:
        six_four_crop = first_stage_crop[0:target, int(lower):int(upper), :]
    return six_four_crop