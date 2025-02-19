#import argparse
import ast
import torch
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt


def run_fastsam_inference(
    model_path="./FastSAM-s.pt",
    input_img=None,
    imgsz=1024,
    iou=0.9,
    conf=0.4,
    point_prompt="[[0,0]]",
    point_label="[1,0]",
    better_quality=False,
    device=None,
    retina=True,
    withContours=False,
):
    """Runs FastSAM inference on a given image.

    Args:
        model_path (str): Path to the FastSAM model.
        input_img: The input image.
        imgsz (int, optional): Image size. Defaults to 1024.
        iou (float, optional): IOU threshold. Defaults to 0.9.
        conf (float, optional): Confidence threshold. Defaults to 0.4.
        output_path (str, optional): Path to save output. Defaults to "./output/".
        point_prompt (str, optional): Point prompt as a string (e.g., "[[350,300]]"). Defaults to "[[0,0]]".
        point_label (str, optional): Point labels (e.g., "[1,0]"). Defaults to "[0]".
        better_quality (bool, optional): Improve quality using morphologyEx. Defaults to False.
        device (str, optional): Device to use ("cuda" or "cpu"). Defaults to auto-selection.
        retina (bool, optional): Use high-resolution segmentation masks. Defaults to True.
        withContours (bool, optional): Draw mask contours. Defaults to False.

    Returns:
        str: Path to the saved segmented image.
    """

    # Auto-detect device if not provided
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    # Load model
    model = FastSAM(model_path)

    # Convert string inputs to actual Python lists
    point_prompt = ast.literal_eval(point_prompt)
    point_label = ast.literal_eval(point_label)

    # Run inference
    results = model(
        input_img,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    # Process prompts
    prompt_process = FastSAMPrompt(input_img, results, device=device)
    
    if point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
    else:
        print("invalid segmentation prompt")

    '''# Get the output image (without the points)
    result = prompt_process.plot_to_result(
        annotations=ann,
        points=None,  # Ensure points are NOT plotted
        point_label=point_label,
        withContours=withContours,
        better_quality=better_quality,
    )'''
    ann = ann.reshape(ann.shape[1], ann.shape[2])
           
    return ann

"""
def parse_args():
    
    # Allowlist SegmentationModel for safe deserialization
    #torch.serialization.add_safe_globals([SegmentationModel])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    '''
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    '''
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    #args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    input = Image.open(args.img_path)
    input = input.convert("RGB")
    everything_results = model(
        input,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou    
        )
    #bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
    
    '''
    if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
    elif args.text_prompt != None:
        ann = prompt_process.text_prompt(text=args.text_prompt)
    el
    '''
    if args.point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=args.point_prompt, pointlabel=args.point_label
        )
        points = args.point_prompt
        point_label = args.point_label
    else:
        ann = prompt_process.everything_prompt()
    prompt_process.plot(
        annotations=ann,
        output_path=args.output+args.img_path.split("/")[-1],
        #bboxes = bboxes,
        points = None, # made this none so that the point will not be plotted
        point_label = point_label,
        withContours=args.withContours,
        better_quality=args.better_quality,
    )




if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
