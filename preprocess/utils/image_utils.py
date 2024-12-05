import numpy as np
import cv2

def clean_mask(mask):
    # remove outliers or isolation points on the mask
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned_mask

def find_point_on_mask(mask, query):
    # get all non-zero point on mask
    points = np.transpose(np.nonzero(mask))
    
    distances = np.linalg.norm(points - query, axis=1)
    nearest_point_index = np.argmin(distances)
    nearest_point = points[nearest_point_index]
    
    return np.array(nearest_point)

def sample_points_inside_contour(origin_mask, contour, n_points=10):
    # Create a mask from the contour
    h, w = origin_mask.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Check if contour needs reshaping
    if len(contour.shape) == 2:
        contour = contour.reshape((-1, 1, 2))
    
    # Draw contour and show counts
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # Get points inside contour
    inside_points = np.column_stack(np.where(mask > 0))
    # print("Number of inside points:", len(inside_points))
    
    if len(inside_points) < n_points:
        return []
        
    sampled_inside = inside_points[np.random.choice(len(inside_points), n_points)]
    points = [(pt[1], pt[0]) for pt in sampled_inside]  # Swap coordinates here
    
    return points

def uniform_sample_points_on_mask(mask, contour, init_sample=1, max_attempts=10):
    height, width = mask.shape
    min_distance = max(height, width) / 20
    cell_size = min_distance / np.sqrt(2)
    
    grid_width = int(width / cell_size)+1
    grid_height = int(height / cell_size)+1
    
    grid = np.zeros((grid_height, grid_width), dtype=np.int32)
    points = []
    active_list = []

    def is_valid_point(x, y):
        is_in_contour = cv2.pointPolygonTest(contour, (x,y), measureDist=False)
        return is_in_contour > 0 and 0 <= x < width and 0 <= y < height and mask[int(y), int(x)] > 0

    def add_point(point):
        points.append(np.array(point, dtype=int))
        active_list.append(point)
        grid_x, grid_y = int(point[0] / cell_size), int(point[1] / cell_size)
        grid[grid_y, grid_x] = len(points)

    # Start with a random point
    while len(points) < init_sample :
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        if is_valid_point(x, y):
            add_point((x, y))
            break

    while active_list:
        idx = np.random.randint(0, len(active_list))
        point = active_list[idx]

        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_distance, 2 * min_distance)
            new_x = point[0] + distance * np.cos(angle)
            new_y = point[1] + distance * np.sin(angle)
            
            if not is_valid_point(new_x, new_y):
                continue

            grid_x, grid_y = int(new_x / cell_size), int(new_y / cell_size)
            if grid[max(0, grid_y-2):min(grid_height, grid_y+3), max(0, grid_x-2):min(grid_width, grid_x+3)].any():
                continue

            add_point((new_x, new_y))
            break
        else:
            active_list.pop(idx)

    return points

def box_contain_pt(box, pt):
    x, y, w, h = box
    if pt[0] < x or pt[1] < y or pt[0] > x+w or pt[1] > y+h:
        return False
    else:
        return True

def union_boxes(box_list):
    x_min = min(bbox[0] for bbox in box_list)
    y_min = min(bbox[1] for bbox in box_list)
    x_max = max(bbox[0] + bbox[2] for bbox in box_list)
    y_max = max(bbox[1] + bbox[3] for bbox in box_list)
    
    union_box = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
    return union_box

def intersect_boxes(box_list):
    x_min = max(bbox[0] for bbox in box_list)
    y_min = max(bbox[1] for bbox in box_list)
    x_max = min(bbox[0] + bbox[2] for bbox in box_list)
    y_max = min(bbox[1] + bbox[3] for bbox in box_list)
    
    if x_max < x_min or y_max < y_min:
        return None
    
    res = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
    return res

def extend_box(box, ratio=0.5):
    x,y,w,h = box
    cx = x + w/2
    cy = y + h/2
    w = (1+ratio) * w
    h = (1+ratio) * h
    x = cx - w/2
    y = cy - h/2
    return np.array([x,y,w,h])
    

def draw_masked_image_with_labels(img, mask, box, points, alpha=0.6, thickness=2):
    # 将掩码应用到图像上
    mask = np.where(mask, 255, 0).astype(np.uint8)

    # 创建一个彩色版本的掩码供叠加使用
    mask_colored = cv2.merge([mask, mask, mask])
    # 生成带有 alpha 值的掩码图像
    masked_img = cv2.addWeighted(img, 1 - alpha, mask_colored, alpha, 0)
    
    # 绘制边界框
    x, y, w, h = box
    cv2.rectangle(masked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(masked_img, 'Box', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness)
    
    # 绘制和标记点
    for idx, point in enumerate(points):
        cv2.circle(masked_img, point, thickness*2, (0, 255, 0), -1)
        cv2.putText(masked_img, f'P{idx}', (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness)
    
    return masked_img

def check_mask_adj_contour(mask, contour_list, thickness=10):
    h, w = mask.shape
    thickness = int(max(h, w) / 100) + 1
    # Step 3: Create boundary image from the given mask
    edges = cv2.Canny(mask, 100, 200)

    # Dilation to make the edge thicker for easier adjacency checking
    kernel = np.ones((9, 9), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Step 4: Check adjacency
    adjacent_contours = []
    omitted_countors = []
    for contour in contour_list:
        omit_flag = True
        for point in contour:
            x, y = point[0]
            if edges_dilated[y, x] > 0:
                adjacent_contours.append(contour)
                omit_flag = False
                break
        if omit_flag:
            omitted_countors.append(contour)
    
    edges_dilated = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2RGB)
    edges_dilated = cv2.bitwise_not(edges_dilated)
    cv2.drawContours(edges_dilated, adjacent_contours, -1, (220, 20, 60), thickness=thickness)  # red for adjacent contour
    cv2.drawContours(edges_dilated, omitted_countors, -1, (50, 205, 50), thickness=thickness)  # green for omitted contour
    
    return adjacent_contours, edges_dilated

def mask_to_sam_prompt(obj_mask, hand_mask, threshold=20):
    H, W = obj_mask.shape
    contour_list, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_list = []
    point_list = []
    
    contour_list, edges_dilated = check_mask_adj_contour(hand_mask, contour_list)
    for contour in contour_list:
        x, y, w, h = cv2.boundingRect(contour)
        center = np.array([x+w//2, y+h//2])
        # if w * h < threshold:
        #     continue
        # nearest_point = find_point_on_mask(obj_mask, center)
        # sampled_point_list = uniform_sample_points_on_mask(obj_mask, contour)
        sampled_point_list = sample_points_inside_contour(obj_mask, contour)
        point_list += sampled_point_list
        
        box_list.append(np.array([x,y,w,h]))
        # if box_contain_pt(box=(x,y,w,h), pt=sampled_point):
        #     point_list.append(sampled_point)
    
    if len(box_list) == 0:
        return None, None, edges_dilated
    box = union_boxes(box_list)  
    box = extend_box(box)  
    box = intersect_boxes([box, np.array([0,0,W-1,H-1])])
    assert box is not None

    box = box.astype(np.int32)
    
    return box, point_list, edges_dilated

def fill_mask(mask):
    mask = np.where(mask, 255, 0).astype(np.uint8)
    # close the boundary
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 方法2: 使用洪水填充法填充孔洞
    # 需要确保掩码边缘是封闭的，否则需要先进行形态学操作
    flood_mask = mask.copy()
    h, w = flood_mask.shape[:2]
    flood_fill_mask = np.zeros((h+2, w+2), np.uint8)

    # 进行洪水填充
    cv2.floodFill(flood_mask, flood_fill_mask, (0, 0), 255)

    # 反转洪水填充结果
    flood_fill_inv = cv2.bitwise_not(flood_mask)

    # 将原始掩码和反转的洪水填充结果合并
    filled_mask = mask | flood_fill_inv
    return filled_mask