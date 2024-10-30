import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from newsapi import NewsApiClient
import threading

class WorldMapApp:
    def __init__(self, master):
        self.master = master
        self.master.title("World Map News")
        
        # Set initial window size
        self.master.geometry("1000x600")
        
        # Initialize NewsAPI client (replace with your actual API key)
        self.newsapi = NewsApiClient(api_key='aebb14142aa744339d441d692e280623')
        
        # Create a canvas to display the image (initialize canvas before loading the map)
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True)

        # Create a frame for news display
        self.news_frame = tk.Frame(self.master, width=300, bg='white')
        self.news_frame.pack(side=tk.RIGHT, fill="y")
        
        # Create a label for news
        self.news_label = tk.Label(self.news_frame, text="Hover over a continent", wraplength=280, justify=tk.LEFT, bg='white')
        self.news_label.pack(padx=10, pady=10)
        
        # Load the world map image and convert it to OpenCV format
        self.load_map()
        
        # Bind mouse motion event
        self.canvas.bind("<Motion>", self.on_hover)
        
    def load_map(self):
        # Load your world map image here (replace with your local path)
        map_path = "C:/Users/rohit/Downloads/1000_F_416348555_iZFArB7fnkG3eFbga3m1bAueHS0x9cpi.jpg"
        
        # Load image using OpenCV and convert it to RGB format for Tkinter display
        self.cv_image = cv2.imread(map_path)
        self.cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV image to PIL format for Tkinter display
        self.pil_image = Image.fromarray(self.cv_image_rgb)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        
        # Display image on canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def on_hover(self, event):
        # Get mouse coordinates relative to the canvas
        x, y = event.x, event.y
        
        if 0 <= x < self.cv_image.shape[1] and 0 <= y < self.cv_image.shape[0]:
            # Get the color of the pixel at the mouse position (in BGR format)
            bgr_color = self.cv_image[y, x]
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

            # Define HSV ranges for each continent's color (you may need to fine-tune these values)
            if self.is_within_hsv_range(hsv_color, (0, 100, 100), (10, 255, 255)):  # Red for North America
                continent = "North America"
            elif self.is_within_hsv_range(hsv_color, (25, 100, 100), (35, 255, 255)):  # Yellow for South America & Asia
                if x < 500:  # Assuming South America is on left side of Asia in your image.
                    continent = "South America"
                else:
                    continent = "Asia"
            elif self.is_within_hsv_range(hsv_color, (110, 50, 50), (130, 255, 255)):  # Blue for Europe & Australia
                if y < 400:  # Assuming Europe is above Australia in your image.
                    continent = "Europe"
                else:
                    continent = "Australia"
            elif self.is_within_hsv_range(hsv_color, (140, 50, 50), (160, 255, 255)):  # Purple for Africa
                continent = "Africa"
            else:
                continent = None
            
            if continent:
                self.fetch_news(continent)

    def is_within_hsv_range(self, hsv_color, lower_bound, upper_bound):
        """Check if a given HSV color falls within a specified range."""
        return all(lower_bound[i] <= hsv_color[i] <= upper_bound[i] for i in range(3))

    def fetch_news(self, continent):
        def fetch():
            try:
                news = self.newsapi.get_top_headlines(category='general', language='en', country='us')
                headlines = [article['title'] for article in news['articles'][:10]]
                news_text = f"Top 10 news for {continent}:\n\n" + "\n\n".join(headlines)
                self.news_label.config(text=news_text)
            except Exception as e:
                self.news_label.config(text=f"Error fetching news: {str(e)}")

        # Run the news fetching in a separate thread to keep the UI responsive.
        threading.Thread(target=fetch, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = WorldMapApp(root)
    root.mainloop()