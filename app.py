import cv2
import numpy as np
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

class ContinentMapApp:
    def __init__(self, image_path, new_width=1500):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        original_height, original_width = self.image.shape[:2]
        aspect_ratio = original_height / original_width

        # Set up the new layout calculations
        self.screen_width = new_width
        self.map_width = int(self.screen_width * 0.75)  # 75% width for the map
        self.news_width = int(self.screen_width * 0.25)  # 25% width for the news
        new_height = int(self.map_width * aspect_ratio)  # Maintain aspect ratio
        self.resized_image = cv2.resize(self.image, (self.map_width, new_height))
        self.current_continent = None
        self.displayed_continent = None
        self.news_fetcher = ContinentNewsFetcher()
        self.news_headlines = {}

    def get_continent_colors(self):
        return {
            'North America': [203, 192, 255],  # Pink (BGR)
            'South America': [0, 255, 255],    # Yellow (BGR)
            'Europe': [255, 204, 153],         # Light Blue (BGR)
            'Africa': [159, 166, 6],           # Purple (BGR)
            'Asia': [0, 204, 0],               # Green (BGR)
            'Australia': [153, 102, 51],       # Blue (BGR)
            'Antarctica': [255, 255, 255]      # White (BGR)
        }

    def check_continent_by_color(self, x, y):
        if x >= self.map_width or y >= self.resized_image.shape[0]:
            return None
        pixel_color = self.resized_image[y, x].tolist()
        continent_colors = self.get_continent_colors()
        closest_continent = None
        min_diff = float('inf')
        for continent_name, color in continent_colors.items():
            diff = np.linalg.norm(np.array(pixel_color) - np.array(color))
            if diff < min_diff:
                min_diff = diff
                closest_continent = continent_name
        return closest_continent

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            new_continent = self.check_continent_by_color(x, y)
            if new_continent != self.current_continent:
                self.current_continent = new_continent
                self.displayed_continent = None  # Clear displayed continent on hover change

        elif event == cv2.EVENT_LBUTTONDOWN:
            clicked_continent = self.check_continent_by_color(x, y)
            if clicked_continent and clicked_continent != 'Antarctica':
                self.displayed_continent = clicked_continent
                if clicked_continent not in self.news_headlines:
                    print(f"Fetching news for {clicked_continent}...")
                    self.news_headlines[clicked_continent] = self.news_fetcher.get_news_for_continent(clicked_continent)
                    print("News fetching complete.")

    def run(self):
        cv2.namedWindow('Continents Map', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Continents Map', self.mouse_event)

        # Load fonts
        title_font = ImageFont.truetype("ARIAL.ttf", 30)  # Adjust path and size as needed
        content_font = ImageFont.truetype("ARIAL.ttf", 16)  # Adjust path and size as needed

        while True:
            canvas = np.ones((self.resized_image.shape[0], self.screen_width, 3), dtype=np.uint8) * 255
            canvas[:, :self.map_width] = self.resized_image

            news_start_x = self.map_width + 10
            news_height = canvas.shape[0] - 20
            news_width = self.news_width - 20

            news_image = Image.new('RGB', (news_width, news_height), color=(240, 240, 240))
            draw = ImageDraw.Draw(news_image)

            if self.current_continent:
                draw.text((20, 20), self.current_continent, font=title_font, fill=(0, 0, 0))

            if self.displayed_continent:
                y_offset = 65
                line_spacing = 25  # Reduced from 30
                headline_spacing = 3  # Reduced from 20
                
                for i, headline in enumerate(self.news_headlines[self.displayed_continent][:10]):
                    # Wrap text
                    wrapped_text = self.wrap_text(headline, content_font, news_width - 40)
                    
                    # Draw bullet point
                    draw.ellipse([20, y_offset + 6, 26, y_offset + 12], fill=(0, 0, 0))
                    
                    # Draw text
                    for j, line in enumerate(wrapped_text):
                        # Add extra space only for the first line of each headline
                        if j == 0:
                            draw.text((40, y_offset), line, font=content_font, fill=(0, 0, 0))
                            y_offset += line_spacing
                        else:
                            draw.text((40, y_offset - 5), line, font=content_font, fill=(0, 0, 0))
                            y_offset += line_spacing - 5
                    
                    y_offset += headline_spacing  # Reduced space between headlines

                    # Check if we're running out of space
                    if y_offset > news_height - 40:
                        draw.text((20, y_offset), "...", font=content_font, fill=(0, 0, 0))
                        break

            news_array = np.array(news_image)
            canvas[10:10+news_height, news_start_x:news_start_x+news_width] = news_array

            cv2.imshow('Continents Map', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Continents Map', cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def wrap_text(self, text, font, max_width):
        lines = []
        words = text.split()
        current_line = words[0]
        for word in words[1:]:
            if font.getsize(current_line + ' ' + word)[0] <= max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines
    

class ContinentNewsFetcher:
    def __init__(self):
        self.rss_feeds = {
            'North America': [
                'http://rssfeeds.usatoday.com/usatoday-NewsTopStories',
                'http://www.npr.org/rss/rss.php?id=1001'
            ],
            'South America': [
                'http://feeds.bbci.co.uk/news/world/latin_america/rss.xml'
            ],
            'Europe': [
                'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
                'https://euobserver.com/rss.xml'
            ],
            'Africa': [
                'https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf',
                'http://feeds.bbci.co.uk/news/world/africa/rss.xml'
            ],
            'Asia': [
                'https://www.channelnewsasia.com/rssfeeds/8395986',
                'http://feeds.bbci.co.uk/news/world/asia/rss.xml'
            ],
            'Australia': [
                'https://www.abc.net.au/news/feed/45910/rss.xml'
            ],
            'Antarctica': [
                'https://antarcticsun.usap.gov/feed/'
            ]
        }

    def fetch_feed(self, url):
        try:
            feed = feedparser.parse(url)
            return [entry.title for entry in feed.entries[:10]]  
        except Exception as e:
            print(f"Error fetching feed {url}: {e}")
            return []

    def get_news_for_continent(self, continent_name):
        feeds = self.rss_feeds.get(continent_name)
        
        headlines = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.fetch_feed, url): url for url in feeds}
            
            for future in as_completed(future_to_url):
                headlines.extend(future.result())

        return headlines if headlines else ["No news found"]

if __name__ == "__main__":
    app = ContinentMapApp('Continents.png')
    app.run()