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

var matrixColorsImg = Array(NUM_PATCH_WIDTH).fill().map(() => Array(NUM_PATCH_WIDTH).fill('')); // cifar image
var matrixColorsAttn = Array(Math.pow(NUM_PATCHES, 2)).fill().map(() => Array(Math.pow(NUM_PATCHES, 2)).fill('')); // attention head

// PLOT image on canvasImg
var patches = JSON.parse(PATCHES_JSON);
var canvasImg = document.getElementById(CANVAS_IMG_ID);
var ctxImg = canvasImg.getContext('2d');
var idx = 0;
for (let i = 0; i < IMAGE_SIZE; i+= PATCH_SIZE) {
    for (let j = 0; j < IMAGE_SIZE; j += PATCH_SIZE) {
        var imgData = ctxImg.createImageData(PATCH_SIZE, PATCH_SIZE);
        var data = imgData.data;
        var patch = patches[idx];

        for (let p = 0, q = 0; p < patch.length; p += 3, q += 4) {
            data[q] = patch[p];
            data[q + 1] = patch[p + 1];
            data[q + 2] = patch[p + 2];
            data[q + 3] = 255;

        }
        const row = Math.floor(i / PATCH_SIZE);
        const col = Math.floor(j / PATCH_SIZE);

        // Storing the representative color for this patch.
        // You can use the first pixel as a representative color, or calculate the average color of the patch.
        matrixColorsImg[row][col] = patch

        ctxImg.putImageData(imgData, j, i);
        ctxImg.strokeStyle = 'white';
        ctxImg.strokeRect(j, i, PATCH_SIZE, PATCH_SIZE);

        idx++;
    }
}

// Plot attention head on canvasAttn
var attnHead = JSON.parse(ATTN_HEAD_JSON);
var canvasAttn = document.getElementById(CANVAS_ATTN_ID);
var ctxAttn = canvasAttn.getContext('2d');
var attnHeadSelector = document.getElementById(ATTN_HEAD_SELECTOR_ID)


// initialize a new attn head
function setNewAttnHead(currentAttnIndex){
    for (let i = 0; i < NUM_PATCHES; i++) {
        for (let j = 0; j < NUM_PATCHES; j++) {
            var color = getColor(attnHead[currentAttnIndex][i][j]);
            ctxAttn.fillStyle = color;
            ctxAttn.fillRect(j * ATTN_SCALING, i * ATTN_SCALING, ATTN_SCALING, ATTN_SCALING);
            matrixColorsAttn[i][j] = color;
        }
    }
}

setNewAttnHead(0);

//// put options in the dropdown box
function populateAttnHeadSelector() {
        attnHead.forEach((_, index) => {
            var option = document.createElement('option');
            option.value = index;
            option.text = 'Attention Head ' + (index + 1);
            attnHeadSelector.appendChild(option);
        });
    }
populateAttnHeadSelector();

// redraw img patch at col row
function redrawImgPatch(col, row) {
        if (matrixColorsImg[row] && matrixColorsImg[row][col]){

        var originalPatch = matrixColorsImg[row][col];
        var imgData = patchToImageData(originalPatch, PATCH_SIZE, PATCH_SIZE);

        ctxImg.putImageData(imgData, col * PATCH_SIZE, row * PATCH_SIZE);
        ctxImg.strokeStyle = 'white';
        ctxImg.strokeRect(col*PATCH_SIZE, row*PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);

        }
}

function redrawImg(){
   for (let r = 0; r < NUM_PATCH_WIDTH; r+= 1) {
        for (let c = 0; c < NUM_PATCH_WIDTH; c += 1) {

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
    return [Math.floor((x - clsTokenOffset) / NUM_PATCH_WIDTH), (x - clsTokenOffset) % NUM_PATCH_WIDTH];
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
    if (event.offsetY == ATTN_SCALING*NUM_PATCHES){
        var x = NUM_PATCHES -1;
    }
    else{
        var x = Math.floor(event.offsetY / ATTN_SCALING);
    }

    const rowColImg = attnPos2ImgRowCol(x);

    if (event.offsetX == ATTN_SCALING*NUM_PATCHES){

        var y = NUM_PATCHES -1;
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
            ctxImg.fillRect(rowColImg[1] * PATCH_SIZE,  rowColImg[0] * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);
        }
        else{
            // CLS Token selected for x, highlight all
            ctxImg.fillStyle = colorTokenA;

          for (let r = 0; r < NUM_PATCH_WIDTH; r+= 1) {
                for (let c = 0; c < NUM_PATCH_WIDTH; c += 1) {

                    ctxImg.fillRect(c * PATCH_SIZE,  r * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);
                }

            }
        }
        if (!(CLS_TOKEN&&(y==0))){
            ctxImg.fillStyle = colorTokenB;
            ctxImg.fillRect(rowColImgSecond[1] * PATCH_SIZE, rowColImgSecond[0] * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);  // Second highlighted pixel
        }
        else{
            // CLS Token selected for y, highlight all
            ctxImg.fillStyle = colorTokenB;

           for (let r = 0; r < NUM_PATCH_WIDTH; r+= 1) {
                for (let c = 0; c < NUM_PATCH_WIDTH; c += 1) {

                    ctxImg.fillRect(c * PATCH_SIZE,  r * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);
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
    lastHighlightedX = null;
    lastHighlightedY = null;
});
