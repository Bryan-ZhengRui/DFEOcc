import torch  

def spiral_scan_indices(n, direction='cw'):  
    """  
    Generate a list of indices for scanning from the center outward in a spiral.  

    Args:  
        n (int): The size of one side of the matrix.  
        direction (str): 'cw' for clockwise scan, 'ccw' for counterclockwise.  

    Returns:  
        torch.LongTensor: A 1D list of indices in the scanned order.  
    """  
    # Ensure n is a positive integer  
    if n <= 0:  
        raise ValueError("Matrix size must be a positive integer")  

    # Initialize result list  
    indices = []  
    # Center of the matrix  
    center = n // 2  
    if n % 2 == 0:  # Even case  
        x, y = center - 1, center - 1  # Start from top-left of center  
    else:  # Odd case  
        x, y = center, center  # Start from the center  

    # Define direction vectors (right, down, left, up) or (left, down, right, up)  
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] if direction == 'cw' else [(0, -1), (1, 0), (0, 1), (-1, 0)]  

    # Current step and direction  
    step = 1  
    direction_index = 0  
    # Add the index of the first point  
    indices.append(x * n + y)  

    while step < n:  
        # Each step repeats twice  
        for _ in range(2):  
            dx, dy = directions[direction_index]  
            for _ in range(step):  
                x += dx  
                y += dy  
                indices.append(x * n + y)  # Directly compute 1D index and add  
            # Switch direction  
            direction_index = (direction_index + 1) % 4  
        # Increase step size  
        step += 1  

    # The last ring (outermost circle)  
    dx, dy = directions[direction_index]  
    for _ in range(step - 1):  
        x += dx  
        y += dy  
        indices.append(x * n + y)  # Directly compute 1D index and add  

    return torch.LongTensor(indices)  


def rearrange_feature_map(feature_map, direction='cw', channel_first=True):  
    """  
    Rearrange the feature map of shape (B, C, H, W) or (B, H, W, C) according to the spiral scan.  

    Args:  
        feature_map (torch.Tensor): Feature map of size (B, C, H, W) or (B, H, W, C).  
        direction (str): 'cw' for clockwise scan, 'ccw' for counterclockwise.  
        channel_first (bool): If True, input is (B, C, H, W); if False, input is (B, H, W, C).  

    Returns:  
        torch.Tensor: Rearranged feature map, shape (B, H * W, C) or (B, C, H * W).  
    """  
    if channel_first:  
        B, C, H, W = feature_map.shape  
        # Generate spiral scan 1D indices  
        indices = spiral_scan_indices(H, direction=direction).to(feature_map.device)  
        # Rearrange directly and flatten to (B, C, H * W)  
        rearranged_feature_map = feature_map.view(B, C, -1)[:, :, indices]  
        return rearranged_feature_map  
    else:  
        B, H, W, C = feature_map.shape    
        # Generate spiral scan 1D indices  
        indices = spiral_scan_indices(H, direction=direction).to(feature_map.device)  
        # Rearrange directly and flatten to (B, H * W, C)  
        rearranged_feature_map = feature_map.view(B, -1, C)[:, indices, :]  
        return rearranged_feature_map   


def restore_feature_map(rearranged_feature_map, direction='cw', channel_first=True):  
    """  
    Restore the original order of the feature map by reversing the spiral rearrangement.  

    Args:  
        rearranged_feature_map (torch.Tensor): Rearranged feature map of size (B, C, H * W) or (B, H * W, C).  
        direction (str): 'cw' for clockwise scan, 'ccw' for counterclockwise.  
        channel_first (bool): If True, input is (B, C, H * W); if False, input is (B, H * W, C).  

    Returns:  
        torch.Tensor: Restored feature map with original shape (B, C, H, W) or (B, H, W, C).  
    """  
    if channel_first:  
        B, C, HW = rearranged_feature_map.shape  
        H = W = int(HW ** 0.5)  # Calculate H and W  
        # Generate spiral scan 1D indices  
        indices = spiral_scan_indices(H, direction=direction).to(rearranged_feature_map.device)  
        # Generate reverse indices to map spiral indices back to original positions  
        reverse_indices = torch.empty_like(indices)  
        reverse_indices[indices] = torch.arange(len(indices), device=rearranged_feature_map.device)  
        # Flatten rearranged feature map to (B * C, H * W)  
        rearranged_flattened = rearranged_feature_map.reshape(B * C, -1)  
        # Restore the original flattened feature map using reverse indices  
        restored_flattened = rearranged_flattened[:, reverse_indices]  
        # Reshape back to (B, C, H, W)  
        restored_feature_map = restored_flattened.view(B, C, H, W)  
        return restored_feature_map  

    else:  
        B, HW, C = rearranged_feature_map.shape  
        H = W = int(HW ** 0.5)  # Calculate H and W  
        # Generate spiral scan 1D indices  
        indices = spiral_scan_indices(H, direction=direction).to(rearranged_feature_map.device)  
        # Generate reverse indices to map spiral indices back to original positions  
        reverse_indices = torch.empty_like(indices)  
        reverse_indices[indices] = torch.arange(len(indices), device=rearranged_feature_map.device)  
        # Flatten rearranged feature map to (B, H * W, C)  
        rearranged_flattened = rearranged_feature_map.reshape(B, -1, C)  
        # Restore the original flattened feature map using reverse indices  
        restored_flattened = rearranged_flattened[:, reverse_indices, :]  
        # Reshape back to (B, H, W, C)  
        restored_feature_map = restored_flattened.view(B, H, W, C)  
        return restored_feature_map  


# example
# B=1
# C=2
# H=5
# W=5 
# # feature_map = torch.randn(B, C, H, W) 
# feature_map = torch.randn(B, H, W, C) 
# print('feature_map',feature_map) 
# rearranged_feature_map = rearrange_feature_map(feature_map, direction='cw',channel_first=False)
# print('-----output-----',rearranged_feature_map)  
# restored_feature_map = restore_feature_map(rearranged_feature_map, direction='cw',channel_first=False)  
# print('-----output2-----',restored_feature_map)  
# assert torch.allclose(feature_map, restored_feature_map) 

