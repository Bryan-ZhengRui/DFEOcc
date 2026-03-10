import torch  

def spiral_scan_indices(n, direction='cw'):  
    """  
    Generate a list of indices for scanning from the center outward in a spiral.  

    Args:  
        n (int): The size of one side of the matrix, must be even.  
        direction (str): 'cw' for clockwise scan, 'ccw' for counterclockwise scan.  

    Returns:  
        torch.LongTensor: A 1D list of indices in the spiral scan order.  
    """  
    if n % 2 != 0:  
        raise ValueError("The size of the matrix must be an even number")  
    # Initialize result list  
    result = []  
    # Center of the matrix  
    center = n // 2  
    if direction == 'cw':  
        x, y = center - 1, center - 1  # Start at the top-left of center  
    elif direction == 'ccw':
        x, y = center - 1, center       # Start at the top-right of center
    if direction == 'cw':  
        # Define direction vectors (right, down, left, up)  
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
    elif direction == 'ccw':  
        # Define direction vectors (left, down, right, up)  
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  
    else:  
        raise ValueError("Direction must be 'cw' or 'ccw'")  
    # Current step and direction  
    step = 1  
    direction_index = 0  
    # Add the first point  
    result.append([x, y])  
    while step < n:  
        # Each step repeats twice  
        for _ in range(2):  
            dx, dy = directions[direction_index]  
            for _ in range(step):  
                x += dx  
                y += dy  
                result.append([x, y])  
            # Switch direction  
            direction_index = (direction_index + 1) % 4  
        # Increase step size  
        step += 1  
    # Final loop (outermost circle)  
    dx, dy = directions[direction_index]  
    for _ in range(step - 1):  
        x += dx  
        y += dy  
        result.append([x, y])  
    # Convert 2D coordinates to 1D indices  
    indices = [i * n + j for i, j in result]  

    return torch.LongTensor(indices)  


def rearrange_feature_map_bchw(feature_map, direction='cw'):  
    """  
    Rearrange a feature map of shape (B, C, H, W) according to spiral scanning order.  

    Args:  
        feature_map (torch.Tensor): Feature map of size (B, C, H, W), H and W must be equal even numbers.  
        direction (str): 'cw' for clockwise scan, 'ccw' for counterclockwise scan.  

    Returns:  
        torch.Tensor: Rearranged feature map with the same shape as input.  
    """  
    B, C, H, W = feature_map.shape  
    if H % 2 != 0 or W % 2 != 0 or H != W:  
        raise ValueError("Height and width of the feature map must be equal even numbers")  
    # Generate spiral scan 1D indices  
    indices = spiral_scan_indices(H, direction=direction).to(feature_map.device)  
    # Flatten the feature map to (B * C, H * W)  
    flattened = feature_map.view(B * C, -1)  
    # Rearrange according to spiral order  
    rearranged_flattened = flattened[:, indices]  
    # Reshape back to (B, C, H, W)  
    rearranged_feature_map = rearranged_flattened.view(B, C, H, W)  

    return rearranged_feature_map  


def restore_feature_map_bchw(rearranged_feature_map, direction='cw'):  
    """  
    Restore the original order of the feature map by reversing the spiral rearrangement.  

    Args:  
        rearranged_feature_map (torch.Tensor): Rearranged feature map of size (B, C, H, W).  
        direction (str): 'cw' for clockwise scan, 'ccw' for counterclockwise scan.  

    Returns:  
        torch.Tensor: Restored feature map with the same shape as input.  
    """  
    B, C, H, W = rearranged_feature_map.shape  
    if H % 2 != 0 or W % 2 != 0 or H != W:  
        raise ValueError("Height and width of the feature map must be equal even numbers")  
    # Generate spiral scan 1D indices  
    indices = spiral_scan_indices(H, direction=direction).to(rearranged_feature_map.device)  
    # Generate reverse indices to map spiral indices back to the original positions  
    reverse_indices = torch.empty_like(indices)  
    reverse_indices[indices] = torch.arange(len(indices), device=rearranged_feature_map.device)  
    # Flatten the rearranged feature map to (B * C, H * W)  
    rearranged_flattened = rearranged_feature_map.reshape(B * C, -1)  
    # Restore the original flattened feature map using reverse indices  
    restored_flattened = rearranged_flattened[:, reverse_indices]  
    # Reshape back to (B, C, H, W)  
    restored_feature_map = restored_flattened.view(B, C, H, W)  

    return restored_feature_map  



# B=1
# C=2
# H=4
# W=4 
# feature_map = torch.randn(B, C, H, W) 
# print('feature_map',feature_map) 
# rearranged_feature_map = rearrange_feature_map_bchw(feature_map, direction='ccw')
# print('-----output-----',rearranged_feature_map)  
# restored_feature_map = restore_feature_map_bchw(rearranged_feature_map, direction='ccw')  
# print('-----output2-----',restored_feature_map)  
# assert torch.allclose(feature_map, restored_feature_map) 

