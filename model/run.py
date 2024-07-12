import numpy as np
import torch
from torch import nn
from tkinter import *
from tkinter import filedialog, simpledialog
from PIL import ImageDraw
import PIL

WIDTH, HEIGHT = 400, 400
CENTER = WIDTH // 2
BLACK = (0, 0, 0)
MODEL_PATH = 'models/MNIST_model_revised.pth'
model_modified = False

d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
device = next(device for device, available in d_opts if available)
print(f'using device: {device}')

def img_to_arr(img_path: str) -> np.ndarray:
    with PIL.Image.open(img_path) as img:
        img = img.convert('RGB').resize((28, 28))
        rgb_arr = np.array(img)
        grayscale_arr = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        grayscale_arr = grayscale_arr.astype(np.uint8)
    return grayscale_arr.flatten()

class mnist_model(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )
        self.block2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )
        self.block3 = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class PaintGUI:
    def __init__(self, model):
        self.model = model

        self.root = Tk()
        self.root.title('Number Classifier')

        self.brush_width = 5
        self.current_color = '#ffffff'

        self.cnv = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg='black')
        self.cnv.pack()
        self.cnv.bind('<B1-Motion>', self.paint)

        self.image = PIL.Image.new('RGB', (WIDTH, HEIGHT), BLACK)
        self.draw = ImageDraw.Draw(self.image)

        self.btn_frame = Frame(self.root)
        self.btn_frame.pack(fill=X)

        self.btn_frame.columnconfigure(0, weight=1)

        self.clear_btn = Button(self.btn_frame, text='Clear', command=self.clear)
        self.clear_btn.grid(row=1, column=0, sticky=W+E)

        self.classify_btn = Button(self.btn_frame, text='Classify', command=self.classify)
        self.classify_btn.grid(row=2, column=0, sticky=W+E)

        self.classify_btn = Button(self.btn_frame, text='Train', command=self.train)
        self.classify_btn.grid(row=3, column=0, sticky=W+E)

        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cnv.create_rectangle(
                x1, y1, x2, y2, 
                outline=self.current_color, 
                fill=self.current_color, 
                width=self.brush_width
        )
        self.draw.rectangle(
                [x1, y1, x2 + self.brush_width, y2 + self.brush_width], 
                outline=self.current_color, 
                fill=self.current_color,
                width=self.brush_width
        )

    def _save(self):
        filename = 'num.png'
        if filename != '':
            self.image.save(filename)

    def _save_to_arr(self) -> torch.Tensor:
        self._save()
        arr = img_to_arr('num.png')
        arr = arr/arr.max()
        arr = torch.from_numpy(arr).type(torch.float).to(device)
        arr = arr.unsqueeze(dim=0)
        return arr

    def classify(self):
        arr = self._save_to_arr()

        self.model.eval()
        with torch.inference_mode():
            pred = self.model(arr)
            pred = torch.softmax(pred, dim=1)
            n = pred.argmax()
            pred = pred.cpu().numpy()
            print(f'=> {n}, probability: {pred[0][n] * 100:.2f}%')

    def _train_single_step(self, arr: torch.Tensor, label: torch.Tensor):
        label = torch.from_numpy(label).to(device)
        loss_fn = loss_fn = nn.CrossEntropyLoss()
        optimizer = optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        out = self.model(arr)
        loss = loss_fn(out, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def train(self):
        global model_modified
        model_modified = True
        arr = self._save_to_arr()

        label = int(simpledialog.askstring('Label', 'Label:'))
        self._train_single_step(arr, np.array(label))
        print(f'Trained network with 1 sample of class: \'{label}\'')

    def clear(self):
        self.cnv.delete('all')
        self.draw.rectangle([0, 0, 1000, 1000], fill='black')

    def on_closing(self):
        if model_modified:
            print(f'Saving model to: {MODEL_PATH}')
            from pathlib import Path
            torch.save(obj=self.model.state_dict(),
                       f=MODEL_PATH)
        print('Closing window')
        self.root.destroy()

if __name__ == '__main__':
    m = mnist_model(n_in=784, n_hidden=128, n_out=10).to(device)
    m.load_state_dict(torch.load(f=MODEL_PATH, map_location=device))

    PaintGUI(m)