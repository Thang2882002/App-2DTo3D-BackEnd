from fastapi import FastAPI, UploadFile, File
import os 
import shutil
import base64
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import api.layer as layer
import torch
import torch.nn as nn
import glob
import cv2
import numpy as np
import os.path as osp
from apps.recon import reconWrapper
import argparse
import uvicorn
import pathlib



from fastapi.responses import JSONResponse
from pathlib import Path
app = FastAPI()
folder_path2d3d = "image"
model2 = ("pifuhd.pt")
folder_out = "image_output"
model1 = ("IMGModel.pth")
device = torch.device('cpu')



# Danh sách các nguồn gốc được phép
origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8000/get_image/",
    "http://localhost:8000/upload_image",
    "http://192.168.1.10:8000/upload_image",
    "http://127.0.0.1:8000/upload_imagehd",
    "http://127.0.0.1:8000/get_imagehd",
    "http://127.0.0.1:8000/get_imagehd_output",
    "http://127.0.0.1:8000/get_image_output",
    "http://127.0.0.1:8000/get_video",
    "http://127.0.0.1:8000/convert",
    "http://127.0.0.1:8000/convert2d3d",
    "http://127.0.0.1:8000/convertvideo",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Danh sách hoặc chuỗi '*' để cho phép tất cả
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức hoặc danh sách cụ thể ['GET','POST',...]
    allow_headers=["*"],  # Cho phép tất cả headers hoặc danh sách cụ thể
)

custom_arg_value = os.environ.get("CUSTOM_ARG", "default_value")


def load2d3dVid():
        # Định nghĩa các tham số mặc định
    default_args = {
        'input_path': "./api/img/image",
        'out_path': "./image_output",
        'ckpt_path': "./pifuhd.pt",
        'resolution': 512,
        'use_rect': False,
        'custom_arg': 'default_value'
    }

    # Tùy chọn tham số dòng lệnh có thể ghi đè tham số mặc định
 

    # Kết hợp các tham số từ cả hai nguồn
    args = {**default_args}

    # Xây dựng lệnh cmd từ các tham số đã được xác định
    start_id = -1
    end_id = -1
    resolution = str(args['resolution'])
    cmd = ['--dataroot', args['input_path'], '--results_path', args['out_path'],
        '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path',
        args['ckpt_path'], '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]

    # Gọi reconWrapper với các tham số tương ứng
    reconWrapper(cmd, args['use_rect'])



def load2d3d():
        # Định nghĩa các tham số mặc định
    default_args = {
        'input_path': "./api/img/image",
        'out_path': "./image_output",
        'ckpt_path': "./pifuhd.pt",
        'resolution': 512,
        'use_rect': False,
        'custom_arg': 'default_value'
    }

    # Tùy chọn tham số dòng lệnh có thể ghi đè tham số mặc định
 

    # Kết hợp các tham số từ cả hai nguồn
    args = {**default_args}

    # Xây dựng lệnh cmd từ các tham số đã được xác định
    start_id = -1
    end_id = -1
    resolution = str(args['resolution'])
    cmd = ['--dataroot', args['input_path'], '--results_path', args['out_path'],
        '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path',
        args['ckpt_path'], '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]

    # Gọi reconWrapper với các tham số tương ứng
    reconWrapper(cmd, args['use_rect'])


def load_modelIMGR():
        model1Load = layer.RRDBNet(3,3,64,23,gc=32)
        model1Load.load_state_dict(torch.load(model1), strict=True)
        model1Load.eval()
        model1Load = model1Load.to(device)
        test_img_folder = 'imagehd/*'
        idx = 0
        for path in glob.glob(test_img_folder):
            idx += 1
            base = osp.splitext(osp.basename(path))[0]
            
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_input = img.unsqueeze(0)
            img_input = img_input.to(device)

            with torch.no_grad():
                output = model1Load(img_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            cv2.imwrite('imagehd_output/{:s}_rst.jpg'.format(base), output)
            print('Convert success')

    

# Một hàm xử lý yêu cầu GET trả về một đối tượng JSON
    
@app.get("/get_image")
async def get_last_image():
    # Đặt đường dẫn đến thư mục chứa ảnh
    image_directoryimg = Path("api/img/image")
    # Lấy danh sách tất cả các file trong thư mục
    image_files = list(image_directoryimg.glob('*'))
    # Kiểm tra xem có file ảnh nào không
    if not image_files:
        return {"error": "No images found in the directory"}
    ## Lấy ảnh cuốitrong danh sách
    last_image_pathimg =max( image_files,key=lambda f:f.stat().st_mtime)
    # Trả về ảnh đầu tiên
    return FileResponse(last_image_pathimg)

@app.get("/get_image_output")
async def get_last_imageo():
    # Đặt đường dẫn đến thư mục chứa ảnh
    image_directoryimgo = Path("image_output/pifuhd_final/recon")
    # Lấy danh sách tất cả các file trong thư mục
    image_files = list(image_directoryimgo.glob('*'))
    # Kiểm tra xem có file ảnh nào không
    if not image_files:
        return {"error": "No images found in the directory"}
    # Lấy ảnh cuốitrong danh sách
    last_image_pathimgo =max( image_files,key=lambda f:f.stat().st_mtime)
    # Trả về ảnh đầu tiên
    return FileResponse(last_image_pathimgo)


    

@app.get("/get_imagehd")
async def get_last_imagehd():
    # Đặt đường dẫn đến thư mục chứa ảnh
    image_directoryhd = Path("imagehd")
    # Lấy danh sách tất cả các file trong thư mục
    image_files = list(image_directoryhd.glob('*'))
    # Kiểm tra xem có file ảnh nào không
    
    if not image_files:
        return {"error": "No images found in the directory"}
    # Lấy ảnh cuốitrong danh sách
    last_image_pathhd =max( image_files,key=lambda f:f.stat().st_mtime)
   
    # Trả về ảnh đầu tiên
   
    return FileResponse(last_image_pathhd)



@app.get("/get_imagehd_output")
async def get_last_imagehdo():
    # Đặt đường dẫn đến thư mục chứa ảnh
    image_directoryhdo = Path("imagehd_output")
    # Lấy danh sách tất cả các file trong thư mục
    image_files = list(image_directoryhdo.glob('*'))
    # Kiểm tra xem có file ảnh nào không
    if not image_files:
        return {"error": "No images found in the directory"}
    # Lấy ảnh cuốitrong danh sách
    last_image_pathhdo =max( image_files,key=lambda f:f.stat().st_mtime)
    # Trả về ảnh đầu tiên
    return FileResponse(last_image_pathhdo)

@app.get("/get_video")
async def get_video():
    # Đặt đường dẫn đến thư mục chứa ảnh
    video_directoryimg = Path("video_output")
    # Lấy danh sách tất cả các file trong thư mục
    video_files = list(video_directoryimg.glob('*'))
    # Kiểm tra xem có file ảnh nào không
    if not video_files:
        return {"error": "No images found in the directory"}
    ## Lấy ảnh cuốitrong danh sách
    last_image_pathimg =max( video_files,key=lambda f:f.stat().st_mtime)
    # Trả về ảnh đầu tiên
    return FileResponse(last_image_pathimg)

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    try:
        for filename in os.listdir('api/img/image'):
            file_path = os.path.join('api/img/image', filename)
            # Kiểm tra xem đó có phải là tệp không
            if filename.endswith('.png') or filename.endswith('.jpg') and os.path.isfile(file_path):
                # Xóa tệp
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        for filename in os.listdir('image_output/pifuhd_final/recon'):
            file_path = os.path.join('image_output/pifuhd_final/recon', filename)
            # Kiểm tra xem đó có phải là tệp không
            if os.path.isfile(file_path):
                # Xóa tệp
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        # Lưu ảnh vào thư mục 'images' với tên file gốc
        with open(f"api/img/image/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        return JSONResponse(content={"message": "Image uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/upload_imagehd")
async def upload_imagehd(image: UploadFile = File(...)):
    try:
        for filename in os.listdir('imagehd'):
            file_path = os.path.join('imagehd', filename)
            # Kiểm tra xem đó có phải là tệp không
            if os.path.isfile(file_path):
                # Xóa tệp
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        for filename in os.listdir('imagehd_output'):
            file_path = os.path.join('imagehd_output', filename)
            # Kiểm tra xem đó có phải là tệp không
            if os.path.isfile(file_path):
                # Xóa tệp
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
       
        # Lưu ảnh vào thư mục 'images' với tên file gốc
        with open(f"imagehd/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        return JSONResponse(content={"message": "Image uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)    
@app.get("/convert")
async def converthd():
    try:
        load_modelIMGR()
        return "success"
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.get("/convert2d3d")
async def convert():
        load2d3d()
        return "success1"

@app.get("/convertvideo")
async def convertvd():
        load2d3dVid()
        return "success1"
    