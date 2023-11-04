## init
import subprocess, os, sys
sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(f"{sub_p_res[:-1]}")
#@markdown **Environment Setup**
import subprocess, time, gc, os, sys

totalFrames = 128
shapeX = 8.0
shapeY = 16.0

def setup_environment():
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    sys.path.extend(['src'])

setup_environment()

import torch
import random
import clip
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts

def PathSetup():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}
    return locals()

root = SimpleNamespace(**PathSetup())
root.models_path, root.output_path = get_model_output_paths(root)

def ModelSetup():
    map_location = "cuda" #@param ["cpu", "cuda"]
    model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
    model_checkpoint =  "v1-5-pruned-emaonly.ckpt" #@param ["custom","dreamlike-photoreal-2.0.ckpt","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = "" #@param {type:"string"}
    return locals()

root.__dict__.update(ModelSetup())
root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

def DeforumAnimArgs():
    animation_mode = 'Video Input' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = totalFrames #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0:(0.0)"#@param {type:"string"}
    translation_x = "0:(0)"#@param {type:"string"}
    translation_y = "0:(0.0)"#@param {type:"string"}
    translation_z = "0:(0.0)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(1)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
    hybrid_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
    hybrid_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}
    enable_schedule_samplers = False #@param {type:"boolean"}
    sampler_schedule = "0:('euler'),10:('dpm2'),20:('dpm2_ancestral'),30:('heun'),40:('euler'),50:('euler_ancestral'),60:('dpm_fast'),70:('dpm_adaptive'),80:('dpmpp_2s_a'),90:('dpmpp_2m')" #@param {type:"string"}
    kernel_schedule = "0: (5)"#@param {type:"string"}
    sigma_schedule = "0: (1.0)"#@param {type:"string"}
    amount_schedule = "0: (0.2)"#@param {type:"string"}
    threshold_schedule = "0: (0.0)"#@param {type:"string"}
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
    color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
    color_force_grayscale = False #@param {type:"boolean"}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}
    video_init_path ='./outputs/2023-09/Pneuma/capture.webm'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}
    hybrid_generate_inputframes = False #@param {type:"boolean"}
    hybrid_use_first_frame_as_init_image = True #@param {type:"boolean"}
    hybrid_motion = "None" #@param ['None','Optical Flow','Perspective','Affine']
    hybrid_motion_use_prev_img = False #@param {type:"boolean"}
    hybrid_flow_method = "DIS Medium" #@param ['DenseRLOF','DIS Medium','Farneback','SF']
    hybrid_composite = False #@param {type:"boolean"}
    hybrid_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_comp_mask_inverse = False #@param {type:"boolean"}
    hybrid_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
    hybrid_comp_mask_auto_contrast = False #@param {type:"boolean"}
    hybrid_comp_save_extra_frames = False #@param {type:"boolean"}
    hybrid_use_video_as_mse_image = False #@param {type:"boolean"}
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 20 #@param {type:"number"}
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}
    return locals()

# prompts
prompts = { 0 : "A photograph of a dark and eerie forest, fog, colorized with colors of a beautiful landscape during sunset, bright moon light and shadow, color, focus, old film photogrpahy, light leaks" }
neg_prompts = { 0 : "dots, lines" }

override_settings_with_file = False #@param {type:"boolean"}
settings_file = "custom" #@param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
custom_settings_file = "/content/drive/MyDrive/Settings.txt"#@param {type:"string"}

def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 256 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    bit_depth_output = 8 #@param [8, 16, 32] {type:"raw"}
    seed = 2859783428 #@param
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 50 #@param
    clamp_steps = 1 #@param
    scale = 7.2 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None
    save_samples = False #@param {type:"boolean"}
    save_settings = False #@param {type:"boolean"}
    display_samples = False #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}
    n_batch = 1 #@param
    n_samples = 1 #@param
    batch_name = "Pneuma" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "fixed" #@param ["iter","fixed","random","ladder","alternate"]
    seed_iter_N = 1 #@param {type:'integer'}
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param
    outdir = get_output_folder(root.output_path, batch_name)
    use_init = True #@param {type:"boolean"}
    strength = 0.8 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "./outputs/2023-08/Tests/render_0.png" #@param {type:"string"}
    add_init_noise = False #@param {type:"boolean"}
    init_noise = 0.01 #@param
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    overlay_mask = True  # {type:"boolean"}
    mask_overlay_blur = 5 # {type:"number"}
    mean_scale = 0 #@param {type:"number"}
    var_scale = 0 #@param {type:"number"}
    exposure_scale = 0 #@param {type:"number"}
    exposure_target = 0.4 #@param {type:"number"}
    colormatch_scale = 0 #@param {type:"number"}
    colormatch_image = "https://cdna.artstation.com/p/assets/images/images/063/574/324/large/athul-krishna-jayan-mobile-wallpapper-test5-min.jpg?1685862627" #@param {type:"string"}
    colormatch_n_colors = 4 #@param {type:"number"}
    ignore_sat_weight = 0 #@param {type:"number"}
    clip_name = 'ViT-L/14' #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0 #@param {type:"number"}
    aesthetics_scale = 0 #@param {type:"number"}
    cutn = 1 #@param {type:"number"}
    cut_pow = 0.0001 #@param {type:"number"}
    init_mse_scale = 0 #@param {type:"number"}
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
    blue_scale = 0 #@param {type:"number"}
    gradient_wrt = 'x0_pred' #@param ["x", "x0_pred"]
    gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
    decode_method = 'linear' #@param ["autoencoder","linear"]
    grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2 #@param {type:"number"}
    clamp_start = 0.2 #@param
    clamp_stop = 0.01 #@param
    grad_inject_timing = list(range(1,10)) #@param
    cond_uncond_sync = True #@param {type:"boolean"}
    precision = 'autocast'
    C = 4
    f = 8
    cond_prompt = ""
    cond_prompts = ""
    uncond_prompt = ""
    uncond_prompts = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0

    return locals()

args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()
args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)
args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    if (args.aesthetics_scale > 0):
        root.aesthetics_model = load_aesthetics_model(args, root)

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None
if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
    print(f"Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = 'klms'
if args.sampler != 'ddim':
    args.ddim_eta = 0

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

gc.collect()
torch.cuda.empty_cache()

cond, uncond = Prompts(prompt=prompts,neg_prompt=neg_prompts).as_dict()

# make sprite
import glob
from PIL import Image
import math, time
def makeSprite( title, listfiles, background, outputdir ):

  frames = []

  tile_width = 0
  tile_height = 0

  dimsX = shapeX
  dimsY = shapeY

  for current_file in listfiles :
    try:
        with Image.open(current_file) as im :
            frames.append(im.getdata())
    except:
        print(current_file + ' is not a valid image')
  
  tile_width = frames[0].size[0]
  tile_height = frames[0].size[1]

  spritesheet_width = tile_width * dimsX
  spritesheet_height = tile_height * dimsY

  spritesheet = Image.new('RGB',(int(spritesheet_width), int(spritesheet_height)),background)

  for current_frame in frames :
      top = tile_height * math.floor((frames.index(current_frame))/dimsX)
      left = tile_width * (frames.index(current_frame) % dimsX)
      bottom = top + tile_height
      right = left + tile_width

      box = (left,top,right,bottom)
      box = [int(i) for i in box]
      cut_frame = current_frame.crop((0,0,tile_width,tile_height))

      spritesheet.paste(cut_frame, box)

  spritesheet.save( os.path.join( outputdir, title + '.png' ) )
  
  for cfile in listfiles :
    os.remove(cfile)

def exportSprite():
    outputdir = "./outputs/" + time.strftime('%Y') + "-" + time.strftime('%m') + "/"+args.batch_name + '/'
    listfiles = glob.glob( "./outputs/" + time.strftime('%Y') + "-" + time.strftime('%m') + "/"+args.batch_name + '/*_*.png')
    listfiles = listfiles[:totalFrames]
    print( " Processing " + str(len(listfiles )) + " frames" )
    makeSprite( "sprite", listfiles, 0, outputdir )



## server
from distutils.log import debug
from fileinput import filename
from flask import Flask, jsonify, request
import base64
import cv2
from flask_cors import CORS
import shutil

working = False
app = Flask(__name__)
CORS(app)    


@app.route('/', methods = ['POST'])
def success():
    count = 1
    global working
    outputdir = "./outputs/" + time.strftime('%Y') + "-" + time.strftime('%m') + "/Pneuma/"
    inpath = os.path.join(outputdir, 'inputframes')
    
    
    if request.method == 'POST':
        if( working == True ) : return
        working = True
        
        for filename in os.listdir(outputdir):
            file_path = os.path.join(outputdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.mkdir(inpath)

        with open(outputdir+'\imageToSave.png', "wb") as fh:
            imgdata = base64.b64decode(request.form['media_data'])
            channel = request.form['channel']
            fh.write(imgdata)
            img = cv2.imread(outputdir+'\imageToSave.png')
            sizeX =  math.floor(img.shape[1] / shapeX )
            sizeY =  math.floor(img.shape[0] / shapeY )
            for r in range(0, img.shape[0], sizeY):
                for c in range(0, img.shape[1], sizeX):
                    cv2.imwrite(outputdir+"inputframes/"+str(count).zfill(5)+".jpg",img[r:r+sizeY,c:c+sizeX,:])
                    count = count+1
        # os.remove(outputdir+'\imageToSave.png')
        args.clamp_steps = int(request.form['steps'])
        prompts[0] = request.form['prompt']
        neg_prompts[0] = request.form['neg_prompt']
        args.strength = float(request.form['strength'])
        args.seed = random.randint(0, 2**32 - 1)
        cond, uncond = Prompts(prompt=prompts,neg_prompt=neg_prompts).as_dict()
        render_animation(root, anim_args, args, cond, uncond)
        exportSprite()
        
        with open(outputdir+"sprite.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            working = False
            return jsonify({"channel": channel ,"sprite": str(encoded_string) })

def run_server_api():
    app.run(host='0.0.0.0', port=8080)
  
  
if __name__ == "__main__":     
    run_server_api()
