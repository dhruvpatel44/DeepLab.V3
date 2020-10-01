# -------- what will I do ? ----------
# ** All you have is trained weights ... which is shit for your brain to understand.
# ** I will pretify your results so you could understand what just happened.
# ------------------------------------

# --------------------------------------
# Neural Network imports
# --------------------------------------
import torch
# --------------------------------------
# pretify imports
# --------------------------------------
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import time

# Load the trained model
model = torch.load("./results/weights.pt")
model.eval()
# print(model)

# Read the log file into a dataframe using pandas
df = pd.read_csv("./results/log.csv")
# print(df.head())

# Training and testing loss, f1_score and auroc values for the model
# ** Plot all the values with respect to the epochs
df.plot(x='epoch',figsize=(15,8))
# plt.show()
plt.savefig('./results/train_test_error.png', bbox_inches='tight')


ino = 40
# Read  a sample image and mask from the data-set
img = cv2.imread(f'./CrackForest/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,480)
mask = cv2.imread(f'./CrackForest/Masks/{ino:03d}_label.PNG')
start = time.time()
with torch.no_grad():
    a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
stop = time.time()

inf_time = stop - start
print(f'Time for inference : {inf_time:.3} seconds')


# Plot the input image, ground truth and the predicted output
plt.figure(figsize=(10,10));
plt.subplot(131);
plt.imshow(img[0,...].transpose(1,2,0));
plt.title('Image')
plt.axis('off');
plt.subplot(132);
plt.imshow(mask);
plt.title('Ground Truth')
plt.axis('off');
plt.subplot(133);
plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.1);
plt.title('Segmentation Output')
plt.axis('off');
# plt.show()
plt.savefig('./results/SegmentationOutput.png',bbox_inches='tight')
