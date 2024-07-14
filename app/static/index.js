const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

const classifyButton = document.getElementById('classify-button');
const clearButton = document.getElementById('clear-button');

const resultElement = document.getElementById('result');

let drawing = false;
let startX;
let startY;

const draw = (e) => {
    if(!drawing) return;

    ctx.lineWidth = 5; 
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvasOffsetX, e.clientY - canvasOffsetY);
    ctx.stroke();
};

async function classifyDigit() {
    const dataUrl = canvas.toDataURL('image/png');
    const response = await fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataUrl })
    });
    const result = await response.json();
    resultElement.textContent = `Prediction: ${result.digit}, Probability: ${result.prob}`;
}

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    startX = e.clientX;
    startY = e.clientY;
});

canvas.addEventListener('mouseup', (e) => {
    drawing = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultElement.textContent = '';
});

classifyButton.addEventListener('click', classifyDigit);