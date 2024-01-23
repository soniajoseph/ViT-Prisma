function patchToImageData(patch, width, height) {
    var imgData = new ImageData(width, height);
    var data = imgData.data;
    for (let p = 0, q = 0; p < patch.length; p += 3, q += 4) {
        data.set([patch[p], patch[p + 1], patch[p + 2], 255], q);
    }
    return imgData;
}