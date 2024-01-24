function getColor(intensity, colorMapName='viridis') {

    const colorMaps = {
        deep: [
            { pos: 0.0, rgb: [76, 114, 176] },
            { pos: 0.1, rgb: [221, 132, 82] },
            { pos: 0.2, rgb: [85, 168, 104] },
            { pos: 0.3, rgb: [196, 78, 82] },
            { pos: 0.4, rgb: [129, 114, 179] },
            { pos: 0.5, rgb: [147, 120, 96] },
            { pos: 0.6, rgb: [218, 139, 195] },
            { pos: 0.7, rgb: [140, 140, 140] },
            { pos: 0.8, rgb: [204, 185, 116] },
            { pos: 0.9, rgb: [100, 181, 205] },
            { pos: 1.0, rgb: [253, 231, 37] } // This last color is not part of seaborn's deep, it's added for completion
        ],
        viridis: [
            {pos: 0, rgb: [68, 1, 84]} ,
            {pos: 0.1, rgb: [72, 34, 115]},
            {pos: 0.2, rgb: [64, 67, 135]},
            {pos: 0.3, rgb: [52, 94, 141]},
            {pos: 0.4, rgb: [41, 120, 142]},
            {pos: 0.5, rgb: [32, 144, 140]},
            {pos: 0.6, rgb: [34, 167, 132]},
            {pos: 0.7, rgb: [68, 190, 112]},
            {pos: 0.8, rgb: [121, 209, 81]},
            {pos: 0.9, rgb: [189, 222, 38]},
            {pos: 1.0, rgb: [253, 231, 37]}
        ]
    };

    // Select the color map based on the colorMapName argument
    const colorMap = colorMaps[colorMapName];

    for (let i = 0; i < colorMap.length - 1; i++) {
        const start = colorMap[i];
        const end = colorMap[i + 1];
        if (intensity >= start.pos && intensity < end.pos) {
            const ratio = (intensity - start.pos) / (end.pos - start.pos);
            const r = Math.floor(start.rgb[0] + ratio * (end.rgb[0] - start.rgb[0]));
            const g = Math.floor(start.rgb[1] + ratio * (end.rgb[1] - start.rgb[1]));
            const b = Math.floor(start.rgb[2] + ratio * (end.rgb[2] - start.rgb[2]));
            return `rgba(${r}, ${g}, ${b}, 1.0)`;
        }
    }
    return `rgba(253, 231, 37, 1.0)`;
}
