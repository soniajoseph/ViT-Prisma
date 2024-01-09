function patchToImageData(patch, width, height) {
    var imgData = new ImageData(width, height);
    var data = imgData.data;
    for (let p = 0, q = 0; p < patch.length; p += 3, q += 4) {
        data[q] = patch[p];
        data[q + 1] = patch[p + 1];
        data[q + 2] = patch[p + 2];
        data[q + 3] = 255;
    }
    return imgData;
}