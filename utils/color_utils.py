import numpy as np

def generate_distinct_colors(num_colors):
    """
    Generate a list of distinct colors from a predefined palette.
    
    Parameters:
    - num_colors: Number of colors to generate
    
    Returns:
    - colors: List of distinct HTML colors
    """
    # Define a palette of distinct colors (colorblind-friendly)
    palette = [
        '#e6194B',  # Red
        '#3cb44b',  # Green
        '#4363d8',  # Blue
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#42d4f4',  # Cyan
        '#f032e6',  # Magenta
        '#bfef45',  # Lime
        '#fabed4',  # Pink
        '#469990',  # Teal
        '#dcbeff',  # Lavender
        '#9A6324',  # Brown
        '#fffac8',  # Beige
        '#800000',  # Maroon
        '#aaffc3',  # Mint
        '#808000',  # Olive
        '#ffd8b1',  # Apricot
        '#000075',  # Navy
        '#a9a9a9',  # Grey
        '#ffffff',  # White
    ]
    
    # If we need more colors than in our palette, we'll create variations
    if num_colors <= len(palette):
        return palette[:num_colors]
    else:
        # Start with our palette
        colors = palette.copy()
        
        # Add more colors by creating variations of the palette
        # We'll adjust brightness and saturation
        remaining = num_colors - len(palette)
        
        # Create darker versions
        for i in range(min(remaining, len(palette))):
            hex_color = palette[i]
            # Convert hex to RGB
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            
            # Make darker (multiply by 0.7)
            r = int(r * 0.7)
            g = int(g * 0.7)
            b = int(b * 0.7)
            
            # Convert back to hex
            dark_color = f'#{r:02x}{g:02x}{b:02x}'
            colors.append(dark_color)
            remaining -= 1
            
            if remaining == 0:
                break
        
        # If we still need more colors, create lighter versions
        for i in range(min(remaining, len(palette))):
            hex_color = palette[i]
            # Convert hex to RGB
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            
            # Make lighter (add 30% of the way to 255)
            r = min(255, int(r + (255 - r) * 0.3))
            g = min(255, int(g + (255 - g) * 0.3))
            b = min(255, int(b + (255 - b) * 0.3))
            
            # Convert back to hex
            light_color = f'#{r:02x}{g:02x}{b:02x}'
            colors.append(light_color)
            remaining -= 1
            
            if remaining == 0:
                break
        
        # If we STILL need more colors, we'll have to resort to random ones
        # but we'll try to make them distinct from existing ones
        while remaining > 0:
            # Generate a random color
            r = np.random.randint(30, 220)  # Avoid extremes for better visibility
            g = np.random.randint(30, 220)
            b = np.random.randint(30, 220)
            new_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Check if it's distinct enough from existing colors
            min_distance = float('inf')
            for color in colors:
                # Calculate color distance in RGB space
                r1 = int(color[1:3], 16)
                g1 = int(color[3:5], 16)
                b1 = int(color[5:7], 16)
                
                distance = np.sqrt((r - r1)**2 + (g - g1)**2 + (b - b1)**2)
                min_distance = min(min_distance, distance)
            
            # If the color is distinct enough, add it
            if min_distance > 100:  # Threshold for "distinct enough"
                colors.append(new_color)
                remaining -= 1
        
        return colors

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to RGBA array"""
    if hex_color.startswith('#'):
        color = hex_color[1:]
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
        a = alpha
        if len(color) >= 8:  # If alpha is included
            a = int(color[6:8], 16) / 255.0
        return [r, g, b, a]
    else:
        # Default color if hex format is invalid
        return [0.7, 0.7, 0.7, alpha]

def generate_colors_for_dark_background(num_colors):
    """
    Generate a list of distinct colors optimized for visibility on dark/black backgrounds.
    
    Parameters:
    - num_colors: Number of colors to generate
    
    Returns:
    - colors: List of distinct HTML colors that are highly visible on dark backgrounds
    """
    # Define a palette of bright, high-contrast colors for dark backgrounds
    dark_bg_palette = [
        '#00FF00',  # Bright Green
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FFFF00',  # Yellow
        '#FF3333',  # Bright Red
        '#33FF33',  # Lime Green
        '#3333FF',  # Bright Blue
        '#FF9933',  # Orange
        '#99FF33',  # Yellow-Green
        '#FF33FF',  # Pink
        '#33FFFF',  # Light Blue
        '#CCFF33',  # Chartreuse
        '#FF6699',  # Rose
        '#99CCFF',  # Sky Blue
        '#FFCC33',  # Gold
        '#66FF99',  # Mint
        '#FF99CC',  # Light Pink
        '#99FFFF',  # Pale Cyan
        '#CCCCFF',  # Lavender
        '#FFFFFF',  # White
    ]
    
    # If we need more colors than in our palette, we'll create variations
    if num_colors <= len(dark_bg_palette):
        return dark_bg_palette[:num_colors]
    else:
        # Start with our palette
        colors = dark_bg_palette.copy()
        
        # Add more colors by creating variations
        remaining = num_colors - len(dark_bg_palette)
        
        # Create medium brightness versions
        for i in range(min(remaining, len(dark_bg_palette))):
            hex_color = dark_bg_palette[i]
            # Convert hex to RGB
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            
            # Adjust brightness (to around 80%)
            r = int(r * 0.8)
            g = int(g * 0.8)
            b = int(b * 0.8)
            
            # Convert back to hex
            medium_color = f'#{r:02x}{g:02x}{b:02x}'
            colors.append(medium_color)
            remaining -= 1
            
            if remaining == 0:
                break
        
        # If we still need more colors, generate distinct ones
        while remaining > 0:
            # Generate a bright random color - higher minimums to ensure visibility
            r = np.random.randint(100, 255)
            g = np.random.randint(100, 255)
            b = np.random.randint(100, 255)
            
            # Ensure at least one channel is very bright (>200)
            while max(r, g, b) < 200:
                if np.random.choice([0, 1, 2]) == 0:
                    r = np.random.randint(200, 255)
                elif np.random.choice([0, 1]) == 0:
                    g = np.random.randint(200, 255)
                else:
                    b = np.random.randint(200, 255)
            
            new_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Check if it's distinct enough from existing colors
            min_distance = float('inf')
            for color in colors:
                # Calculate color distance in RGB space
                r1 = int(color[1:3], 16)
                g1 = int(color[3:5], 16)
                b1 = int(color[5:7], 16)
                
                distance = np.sqrt((r - r1)**2 + (g - g1)**2 + (b - b1)**2)
                min_distance = min(min_distance, distance)
            
            # If the color is distinct enough, add it
            if min_distance > 120:  # Higher threshold for more distinction
                colors.append(new_color)
                remaining -= 1
        
        return colors 