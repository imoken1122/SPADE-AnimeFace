import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import lycon
def preprocess(name):
    path_l = glob.glob(f"./data/{name}/full/*")
    print(path_l)

    def preprocess(p):
        img = cv2.resize(lycon.load(p),(256,256))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV )
        img = cv2.medianBlur(img,9)
        tf = cv2.pyrMeanShiftFiltering(img,52,52)
        tf = cv2.medianBlur(tf,15)
        tf = cv2.cvtColor(tf, cv2.COLOR_HSV2BGR)
        p = p.split("/")[-1]
        cv2.imwrite(f"data/{name}/ref/{p}",tf)


    from concurrent.futures  import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        mapgen = list(tqdm( executor.map(preprocess,path_l),total=len(path_l)))

if __name__ == '__main__':
    preprocess("train")
    preprocess("test")