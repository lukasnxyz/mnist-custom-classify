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
MODEL_PATH = "models/MNIST_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_modified = False

def img_to_arr(img_path: str) -> np.ndarray:
    with PIL.Image.open(img_path) as img:
        img = img.convert("RGB").resize((28, 28))
        rgb_arr = np.array(img)
        grayscale_arr = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        grayscale_arr = grayscale_arr.astype(np.uint8)
    return grayscale_arr.flatten()

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784, 128)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(128, 10)
        self.act_output = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.act1(self.h1(X))
        X = self.act_output(self.output(X))
        return X

class PaintGUI:
    def __init__(self, model):
        self.model = model

        self.root = Tk()
        self.root.title("Number Classifier")

        self.brush_width = 5
        self.current_color = "#ffffff"

        self.cnv = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="black")
        self.cnv.pack()
        self.cnv.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("RGB", (WIDTH, HEIGHT), BLACK)
        self.draw = ImageDraw.Draw(self.image)

        self.btn_frame = Frame(self.root)
        self.btn_frame.pack(fill=X)

        self.btn_frame.columnconfigure(0, weight=1)

        self.clear_btn = Button(self.btn_frame, text="Clear", command=self.clear)
        self.clear_btn.grid(row=1, column=0, sticky=W+E)

        self.classify_btn = Button(self.btn_frame, text="Classify", command=self.classify)
        self.classify_btn.grid(row=2, column=0, sticky=W+E)

        self.classify_btn = Button(self.btn_frame, text="Train", command=self.train)
        self.classify_btn.grid(row=3, column=0, sticky=W+E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
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
        filename = "num.png"
        if filename != "":
            self.image.save(filename)

    def _save_to_arr(self) -> torch.Tensor:
        self._save()
        arr = img_to_arr("num.png")
        arr = arr/arr.max()
        arr = torch.from_numpy(arr).type(torch.float).to(device)
        return arr

    def classify(self):
        arr = self._save_to_arr()

        self.model.eval()
        with torch.inference_mode():
            pred = self.model(arr).numpy()
            n = pred.argmax()
            print(f"=> {n}, probability: {pred[n] * 100:.2f}%")

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

        label = int(simpledialog.askstring("Label", "Label:"))
        self._train_single_step(arr, np.array(label))
        print(f"Trained network with 1 sample of class: \"{label}\"")

    def clear(self):
        self.cnv.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="black")

    def on_closing(self):
        if model_modified:
            print(f"Saving model to: {MODEL_PATH}")
            from pathlib import Path
            torch.save(obj=self.model.state_dict(),
                       f=MODEL_PATH)
        print("Closing window")
        self.root.destroy()

def main():
    print(f"Using device: \"{device}\"")
    model = MNIST().to(device)
    model.load_state_dict(torch.load(f=MODEL_PATH, map_location=device))

    PaintGUI(model)

if __name__ == "__main__":
  main()
