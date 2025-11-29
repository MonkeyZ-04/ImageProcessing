import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ Image Preprocessing ------------------ #
def smooth(image, blur_strength=15):
    # Ensure kernel size is odd and at least 1
    k_size = int(blur_strength)
    if k_size % 2 == 0:
        k_size += 1
    if k_size < 1:
        k_size = 1
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    image_float = clahe_image.astype(np.float32)
    
    # Use the user-defined k_size for the main blurs
    smoothed = cv2.GaussianBlur(image_float, (k_size, k_size), 4.0)
    
    # We scale the secondary blur slightly smaller than the main one
    k_size_secondary = max(3, k_size - 4) 
    if k_size_secondary % 2 == 0: k_size_secondary += 1
        
    smoothed = cv2.GaussianBlur(smoothed, (k_size_secondary, k_size_secondary), 3.0)
    smoothed_uint8 = smoothed.astype(np.uint8)
    
    smoothed = cv2.medianBlur(smoothed_uint8, k_size) # Use user k_size here too
    smoothed = cv2.medianBlur(smoothed, k_size_secondary)
    
    # Bilateral filter helps keep edges sharp while smoothing noise
    smoothed = cv2.bilateralFilter(smoothed, 31, 200, 200)
    
    # Final smoothing
    final_k = max(3, k_size - 6)
    if final_k % 2 == 0: final_k += 1
    smoothed = cv2.GaussianBlur(smoothed.astype(np.float32), (final_k, final_k), 2.5)
    
    return smoothed.astype(np.uint8), clahe_image  

def crop(input_image, kernel_size=15, target_ratio=3.0/2.0):
    if isinstance(input_image, str):
        gray_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            raise ValueError(f"Cannot load image: {input_image}")
    else:
        gray_image = input_image.copy()
    
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lung_mask = np.zeros_like(binary)
    
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(lung_mask, [contour], -1, 255, -1)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    connected_lungs = cv2.dilate(lung_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(connected_lungs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: 
        return gray_image
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    img_h, img_w = gray_image.shape
    padding_x, padding_y = int(w * 0.25), int(h * 0.25)
    bbox_w, bbox_h = w + 2 * padding_x, h + 2 * padding_y
    current_ratio = bbox_w / bbox_h
    
    if current_ratio > target_ratio:
        target_h = bbox_w / target_ratio
        padding_y = max(padding_y, int((target_h - h) / 2))
    else:
        target_w = bbox_h * target_ratio
        padding_x = max(padding_x, int((target_w - w) / 2))
    
    x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
    x2, y2 = min(img_w, x + w + padding_x), min(img_h, y + h + padding_y)
    
    return gray_image[y1:y2, x1:x2]

# ------------------ Dark Region Detection ------------------ #
def find_dark(cropped_image, smoothed_image):
    _, binary = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dark_regions = []
    region_masks = []
    
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(contour)
                if area > 300:
                    dark_regions.append(contour)
                    mask = np.zeros_like(cropped_image)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    region_masks.append(mask)
    return dark_regions, region_masks

# ------------------ Merge Regions ------------------ #
def merge(dark_regions, region_masks, cropped_image, trim_margin=0):
    if not dark_regions:
        return None
    
    largest_idx = np.argmax([cv2.contourArea(contour) for contour in dark_regions])
    largest_region = dark_regions[largest_idx]
    largest_mask = region_masks[largest_idx]
    
    lx, ly, lw, lh = cv2.boundingRect(largest_region)
    largest_center_x = lx + lw // 2
    left_regions = []
    all_target_points = largest_region.reshape(-1, 2) 

    for i, region in enumerate(dark_regions):
        if i == largest_idx: continue
        x, y, w, h = cv2.boundingRect(region)
        region_center_x = x + w // 2
        if region_center_x < largest_center_x:
            left_regions.append(region)
            all_target_points = np.concatenate((all_target_points, region.reshape(-1, 2)), axis=0)
    
    if all_target_points.size == 0:
        return None

    hull = cv2.convexHull(all_target_points)
    initial_merged_mask = np.zeros_like(cropped_image, dtype=np.uint8)
    cv2.drawContours(initial_merged_mask, [hull], 0, 255, -1) 

    height, width = cropped_image.shape
    largest_right_boundary = np.zeros(height, dtype=np.int32) 
    largest_y_coords, largest_x_coords = np.where(largest_mask == 255)
    
    max_x_by_y = {}
    for y, x in zip(largest_y_coords, largest_x_coords):
        max_x_by_y[y] = max(max_x_by_y.get(y, 0), x)
    
    # Apply User Trim Margin Here
    for y, max_x in max_x_by_y.items():
        largest_right_boundary[y] = max_x + trim_margin 

    trimmed_mask = initial_merged_mask.copy()
    for y in range(height):
        boundary_x = largest_right_boundary[y]
        if boundary_x > 0:
            merged_x_coords = np.where(initial_merged_mask[y, :] == 255)[0]
            # Logic: If pixel X is to the right of boundary, remove it (set to 0)
            pixels_to_trim_x = merged_x_coords[merged_x_coords > boundary_x]
            trimmed_mask[y, pixels_to_trim_x] = 0
            
    return largest_mask, trimmed_mask, left_regions

# ------------------ Post-processing ------------------ #
def postprocess_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    mask_smooth = cv2.GaussianBlur(mask_opened, (5,5), 0)
    return mask_smooth.astype(np.uint8)

def calculate_masked_average(cropped_image, mask):
    binary_mask = (mask > 0).astype(np.uint8)
    masked_data = cropped_image * binary_mask
    roi_pixels = masked_data[masked_data > 0]
    if roi_pixels.size == 0:
        return 0.0
    return np.mean(roi_pixels)

# ------------------ Gradient Region Application ------------------ #
def region_gradient(preprocessed_img, region_mask, min_factor=0.0, max_factor=1.0, linear_ratio=0.2):
    mask_bin = (region_mask > 0).astype(np.uint8)
    if np.sum(mask_bin) == 0:
        return preprocessed_img.copy()
    
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist == 0:
        return preprocessed_img.copy()
    
    norm_dist = dist / max_dist
    factor = np.zeros_like(norm_dist, dtype=np.float32)
    linear_threshold = linear_ratio
    linear_mask = norm_dist <= linear_threshold
    expo_mask = norm_dist > linear_threshold
    
    factor[linear_mask] = max_factor - (max_factor - (min_factor + 0.1*(max_factor-min_factor))) * (norm_dist[linear_mask]/linear_threshold)
    k = 10
    factor[expo_mask] = (min_factor + 0.1*(max_factor-min_factor)) * np.exp(-k*(norm_dist[expo_mask]-linear_threshold)/(1-linear_threshold))
    
    factor = np.clip(factor, min_factor, max_factor)
    result = preprocessed_img.astype(np.float32).copy()
    result[mask_bin == 1] *= factor[mask_bin == 1]
    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------ Visualization  ------------------ #
def visualize_batch(results_list):
    n_samples = len(results_list)
    if n_samples == 0:
        print("No results to visualize.")
        return

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    if n_samples == 1:
        axes = np.array([axes])

    for i, (name, cropped, mask, gradient) in enumerate(results_list):
        # Column 1: Cropped Image
        axes[i, 0].imshow(cropped, cmap='gray')
        axes[i, 0].set_title(f"{name}: Cropped")
        axes[i, 0].axis('off')
        
        # Column 2: Mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"{name}: Mask")
        axes[i, 1].axis('off')
        
        # Column 3: Gradient Result
        axes[i, 2].imshow(gradient, cmap='gray')
        axes[i, 2].set_title(f"{name}: Gradient")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# ------------------ Run Full Pipeline (แก้ไขให้ Return ค่า) ------------------ #
def run_pipeline(image_path, user_blur, user_trim, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(image_path).split('.')[0]
    
    try:
        cropped_img = crop(image_path)
    except Exception as e:
        print(e)
        return None

    # PASS USER BLUR HERE
    preprocessed_img, clahe_img = smooth(cropped_img, blur_strength=user_blur)
    
    dark_regions, region_masks = find_dark(cropped_img, preprocessed_img)
    
    if not dark_regions:
        print(f"No dark regions found in {image_path}.")
        return None
    
    # PASS USER TRIM HERE
    merge_result = merge(dark_regions, region_masks, cropped_img, trim_margin=user_trim)
    
    if merge_result is None:
        print(f"Merge failed for {image_path}.")
        return None
    
    largest_mask, merged_mask, left_regions = merge_result
    merged_mask_post = postprocess_mask(merged_mask)
    
    avg_intensity = calculate_masked_average(cropped_img, merged_mask_post)
    print(f"Average intensity in merged region for {image_path}: {avg_intensity:.2f}")
    
    # Apply gradient inside region only
    gradient_img = region_gradient(cropped_img, merged_mask_post)
    
    cv2.imwrite(os.path.join(output_dir, f"{base}_cropped.png"), cropped_img)
    cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), merged_mask_post)
    cv2.imwrite(os.path.join(output_dir, f"{base}_region_gradient.png"), gradient_img)
    
    return (base, cropped_img, merged_mask_post, gradient_img)

# ------------------ Example Usage (แก้ไข Loop) ------------------ #
if __name__ == "__main__":
    image_paths = ['BTR022.jpg', 'BTR024.jpg', 'BTR056.jpg', 'BTR065.jpg']
    all_results = []

    print("--- Settings Configuration ---")
    
    # 1. Ask for Blur Strength
    try:
        blur_input = input("Enter Blur Strength (Default 15, Higher = Smoother): ")
        user_blur = int(blur_input) if blur_input.strip() else 15
    except ValueError:
        print("Invalid input, defaulting to 15.")
        user_blur = 15

    # 2. Ask for Trim Margin
    try:
        trim_input = input("Enter Trim Margin (Default 0, Negative = Cut more, Positive = Expand): ")
        user_trim = int(trim_input) if trim_input.strip() else 0
    except ValueError:
        print("Invalid input, defaulting to 0.")
        user_trim = 0

    print(f"\nProcessing with Blur: {user_blur} | Trim Margin: {user_trim}\n")

    # Run loop with user values
    for path in image_paths:
        if os.path.exists(path):
            result = run_pipeline(path, user_blur, user_trim)
            if result is not None:
                all_results.append(result)
        else:
            print(f"Image not found: {path}")

    print("Visualizing results...")
    visualize_batch(all_results)