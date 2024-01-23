var colorTokenA = 'rgba(0, 128, 128, 0.8)'; //teal
var colorTokenB = 'rgba(255, 105, 180, 0.7)'; //pink

var lastHighlightedX = null;
var lastHighlightedY = null;
var isEntireImageHighlighted = false;

if (CLS_TOKEN) {
    var clsTokenOffset = 1;
} else {
    var clsTokenOffset = 0;
}




var matrixColorsAttn;  // attention head
var matrixColorsImg;  // image
var numPatches;
var patchSize;
var numPatchWidth;

// Plot attention head on canvasAttn
var attnHead = JSON.parse(ATTN_HEAD_JSON);
var canvasAttn = document.getElementById(CANVAS_ATTN_ID);
var ctxAttn = canvasAttn.getContext('2d');
var attnHeadSelector = document.getElementById(ATTN_HEAD_SELECTOR_ID)


// initialize a new attn head
function setNewAttnHead(currentAttnIndex){
    numPatches = attnHead[currentAttnIndex].length;
    canvasAttn.width = numPatches * ATTN_SCALING;
    canvasAttn.height = numPatches * ATTN_SCALING;
    canvasAttn.style.width = numPatches * ATTN_SCALING + "px";
    canvasAttn.style.height = numPatches * ATTN_SCALING + "px";
    matrixColorsAttn = Array(Math.pow(numPatches, 2)).fill().map(() => Array(Math.pow(numPatches, 2)).fill(''));
    for (let i = 0; i < numPatches; i++) {
        for (let j = 0; j < numPatches; j++) {
            var color = getColor(attnHead[currentAttnIndex][i][j]);
            ctxAttn.fillStyle = color;
            ctxAttn.fillRect(j * ATTN_SCALING, i * ATTN_SCALING, ATTN_SCALING, ATTN_SCALING);
            matrixColorsAttn[i][j] = color;
        }
    }
}

setNewAttnHead(0);

var patches = JSON.parse(PATCHES_JSON);
var canvasImg = document.getElementById(CANVAS_IMG_ID);
var ctxImg = canvasImg.getContext('2d');
var imageSizes = JSON.parse(IMAGE_SIZES_JSON);

// initialize a new image
function setNewImage(currentImageIndex){
    var imageSize = imageSizes[currentImageIndex];

    canvasImg.width = imageSize;
    canvasImg.height = imageSize;
    canvasImg.style.width = (imageSize + 20) + "px";
    canvasImg.style.height = (imageSize + 20) + "px";

    patchSize = Math.floor(imageSize / Math.sqrt(numPatches - 1));
    numPatchWidth = Math.floor(imageSize / (patchSize - 1))
    matrixColorsImg = Array(numPatchWidth).fill().map(() => Array(numPatchWidth).fill(''));
    var idx = 0;
    for (let i = 0; i < imageSize; i+= patchSize) {
        for (let j = 0; j < imageSize; j += patchSize) {
            var imgData = ctxImg.createImageData(patchSize, patchSize);
            var data = imgData.data;
            var patch = patches[currentImageIndex][idx];

            for (let p = 0, q = 0; p < patch.length; p += 3, q += 4) {
                data.set([patch[p], patch[p + 1], patch[p + 2], 255], q);
            }
            const row = Math.floor(i / patchSize);
            const col = Math.floor(j / patchSize);

            // Storing the representative color for this patch.
            // You can use the first pixel as a representative color, or calculate the average color of the patch.
            matrixColorsImg[row][col] = patch

            ctxImg.putImageData(imgData, j, i);
            ctxImg.strokeStyle = 'white';
            ctxImg.strokeRect(j, i, patchSize, patchSize);

            idx++;
        }
    }
}
setNewImage(0);

var names = JSON.parse(NAMES_JSON);

//// put options in the dropdown box
function populateAttnHeadSelector() {
        attnHead.forEach((_, index) => {
            var option = document.createElement('option');
            option.value = index;
            option.text = (index + 1) + ": " + names[index];
            attnHeadSelector.appendChild(option);
        });
    }
populateAttnHeadSelector();

// redraw img patch at col row
function redrawImgPatch(col, row) {
        if (matrixColorsImg[row] && matrixColorsImg[row][col]){

        var originalPatch = matrixColorsImg[row][col];
        var imgData = patchToImageData(originalPatch, patchSize, patchSize);

        ctxImg.putImageData(imgData, col * patchSize, row * patchSize);
        ctxImg.strokeStyle = 'white';
        ctxImg.strokeRect(col*patchSize, row*patchSize, patchSize, patchSize);

        }
}

function redrawImg(){
   for (let r = 0; r < numPatchWidth; r+= 1) {
        for (let c = 0; c < numPatchWidth; c += 1) {

            redrawImgPatch(c, r);
        }
   }
}

// redraw attention matrix at x y
function redrawAttnEntry(x,y){
    ctxAttn.fillStyle = matrixColorsAttn[x][y];
    ctxAttn.fillRect(y * ATTN_SCALING, x * ATTN_SCALING, ATTN_SCALING, ATTN_SCALING);
}


// convert row OR col of attention matrix to the corresponding patch row and col
function attnPos2ImgRowCol(x){
    return [Math.floor((x - clsTokenOffset) / numPatchWidth), (x - clsTokenOffset) % numPatchWidth];
}

// redraw everything drawn when x and y are highlighted
function redrawAll(x,y){

    if (y !== null && x !== null){
            if (!(CLS_TOKEN&&x==0))
            {
            const prevrowcolImg = attnPos2ImgRowCol(x);
            redrawImgPatch(prevrowcolImg[1], prevrowcolImg[0]);
            }
            else{
                redrawImg();
            }

            if (!(CLS_TOKEN&&y==0))
            {
            const prevrowcolImgSecond = attnPos2ImgRowCol(y);
            redrawImgPatch(prevrowcolImgSecond[1], prevrowcolImgSecond[0]);
            }
            else{
                redrawImg();
            }

            redrawAttnEntry(x,y);
    }
}


// Add listeners for highlighted pixels
canvasAttn.addEventListener('mousemove', function(event) {

    // find x,y + corresponding img row/cols for the current mouse position
    if (event.offsetY == ATTN_SCALING*numPatches){
        var x = numPatches -1;
    }
    else{
        var x = Math.floor(event.offsetY / ATTN_SCALING);
    }

    const rowColImg = attnPos2ImgRowCol(x);

    if (event.offsetX == ATTN_SCALING*numPatches){

        var y = numPatches -1;
    }
    else{
        var y = Math.floor(event.offsetX / ATTN_SCALING);
    }

    const rowColImgSecond = attnPos2ImgRowCol(y);

    if (lastHighlightedX !== x || lastHighlightedY !== y){

        /// redraw as needed
        redrawAll(lastHighlightedX, lastHighlightedY);

        //draw new highlights

        if (!(CLS_TOKEN&&(x==0))){
            ctxImg.fillStyle = colorTokenA;
            ctxImg.fillRect(rowColImg[1] * patchSize,  rowColImg[0] * patchSize, patchSize, patchSize);
        }
        else{
            // CLS Token selected for x, highlight all
            ctxImg.fillStyle = colorTokenA;

          for (let r = 0; r < numPatchWidth; r+= 1) {
                for (let c = 0; c < numPatchWidth; c += 1) {

                    ctxImg.fillRect(c * patchSize,  r * patchSize, patchSize, patchSize);
                }

            }
        }
        if (!(CLS_TOKEN&&(y==0))){
            ctxImg.fillStyle = colorTokenB;
            ctxImg.fillRect(rowColImgSecond[1] * patchSize, rowColImgSecond[0] * patchSize, patchSize, patchSize);  // Second highlighted pixel
        }
        else{
            // CLS Token selected for y, highlight all
            ctxImg.fillStyle = colorTokenB;

           for (let r = 0; r < numPatchWidth; r+= 1) {
                for (let c = 0; c < numPatchWidth; c += 1) {

                    ctxImg.fillRect(c * patchSize,  r * patchSize, patchSize, patchSize);
                }
           }
        }

        // draw white square on attn matrix
        ctxAttn.fillStyle = 'white';
        ctxAttn.fillRect(y * ATTN_SCALING, x * ATTN_SCALING, ATTN_SCALING, ATTN_SCALING);


    }



    lastHighlightedX = x;
    lastHighlightedY = y;  //


}, { passive: true });

canvasAttn.addEventListener('mouseout', function() {

    redrawAll(lastHighlightedX, lastHighlightedY);

    lastHighlightedX = null;
    lastHighlightedY = null;  // Reset this too

}, { passive: true });


attnHeadSelector.addEventListener('change', function() {
    var selectedIndex = parseInt(this.value);
    setNewAttnHead(selectedIndex);
    setNewImage(selectedIndex);
    lastHighlightedX = null;
    lastHighlightedY = null;
});
