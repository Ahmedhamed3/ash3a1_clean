import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from landmark_infer import load_model, predict


def overlay_landmarks(img, preds, targets):
    """Return a Matplotlib figure with predicted and ground truth landmarks."""
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    preds = np.array(preds).reshape(-1, 2)
    targets = np.array(targets).reshape(-1, 2)
    ax.scatter(-targets[:, 0], -targets[:, 1], c='lime', marker='o', label='GT')
    ax.scatter(-np.array(preds)[:, 0], -np.array(preds)[:, 1], c='red', marker='x', label='Pred')
    ax.legend()
    ax.axis('off')
    return fig


def main(csv_path='dataset/all_19_samples_landmarks_clean.csv',
         img_dir='dataset/xray_images',
         weights='landmark_model.pth',
         output_dir=None, show=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(weights, device)
    data = pd.read_csv(csv_path)
    errors = []
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for _, row in data.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        img = Image.open(img_path).convert('RGB')
        gt = [row['MF_x'], row['MF_y'],
              row['Apex_of_PremolarMolar_x'], row['Apex_of_PremolarMolar_y'],
              row['IAC_x'], row['IAC_y'],
              row['LBM_90_x'], row['LBM_90_y']]
        preds = predict(img_path, model, device)

        # euclidean distances per landmark
        gt_pts = np.array(gt).reshape(-1, 2)
        pred_pts = np.array(preds).reshape(-1, 2)
        dists = np.linalg.norm(pred_pts - gt_pts, axis=1)
        errors.append(dists.mean())

        fig = overlay_landmarks(img, preds, gt)
        if output_dir:
            out_path = os.path.join(output_dir, os.path.splitext(row['filename'])[0] + '.png')
            fig.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    print(f'Average landmark error: {np.mean(errors):.2f} pixels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize landmark predictions')
    parser.add_argument('--csv', default='dataset/all_19_samples_landmarks_clean.csv')
    parser.add_argument('--img_dir', default='dataset/xray_images')
    parser.add_argument('--weights', default='landmark_model.pth')
    parser.add_argument('--output_dir', default=None, help='Directory to save annotated images')
    parser.add_argument('--show', action='store_true', help='Display each overlay')
    args = parser.parse_args()

    main(csv_path=args.csv, img_dir=args.img_dir, weights=args.weights,
         output_dir=args.output_dir, show=args.show)