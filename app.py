
import cv2
import numpy as np
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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
        self.selected_continent = None
        self.news_fetcher = ContinentNewsFetcher()
        self.news_headlines = {}
        self.fetch_all_news()

        # Initialize MediaPipe FaceMesh for eye tracking and blink detection
        # self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        # self.blink_threshold = 0.2  # Threshold for detecting a blink based on EAR
        # self.right_eye_blink_detected = False

    def fetch_all_news(self):
        print("Fetching news for all continents...")
        for continent in self.get_continent_colors().keys():
            if continent != 'Antarctica':
                self.news_headlines[continent] = self.news_fetcher.get_news_for_continent(continent)
        print("News fetching complete.")

    def get_continent_colors(self):
        return {
            'North America': [203, 192, 255],  # Pink (BGR)
            'South America': [0, 255, 255],    # Yellow (BGR)
            'Europe': [255, 204, 153],         # Light Blue (BGR)
            'Africa': [159, 166, 6],           # Purple (BGR)
            'Africa': [159,167,7],         # Teal/Cyan-green (BGR)
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
            continent = self.check_continent_by_color(x, y)
            if continent:
                self.current_continent = continent  
            else:
                self.current_continent = None  

        elif event == cv2.EVENT_LBUTTONDOWN:  
            continent = self.check_continent_by_color(x, y)  
            if continent and continent != 'Antarctica':  
                self.selected_continent = continent  
            else:
                self.selected_continent = None  





    def run(self):
        cv2.namedWindow('Continents Map', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Continents Map', self.mouse_event)

        while True:
            canvas = np.ones((self.resized_image.shape[0], self.screen_width, 3), dtype=np.uint8) * 255
            canvas[:, :self.map_width] = self.resized_image

            news_start_x = self.map_width + 10
            cv2.rectangle(canvas, (news_start_x, 10), (news_start_x + self.news_width - 10, canvas.shape[0] - 10), (240, 240, 240), -1)

            if self.current_continent:
                font_scale, font_thickness, font = 1.2, 2, cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, self.current_continent, (news_start_x + 20, 60), 
                            font, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness, lineType=cv2.LINE_AA)

            if self.selected_continent:
                line_y_offset = 120
                max_lines = 5
                font = cv2.FONT_HERSHEY_SIMPLEX
                for i, headline in enumerate(self.news_headlines[self.selected_continent][:max_lines]):
                    words = headline.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if cv2.getTextSize(current_line + " " + word, font, fontScale=0.6, thickness=1)[0][0] < (self.news_width - 40):
                            current_line += " " + word if current_line else word
                        else:
                            lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)

                    for j, line in enumerate(lines):
                        y_pos = line_y_offset + (i * 80) + (j * 30)
                        if y_pos < canvas.shape[0] - 40:
                            cv2.putText(canvas, f"{i+1}. {line}" if j == 0 else f" {line}", 
                                        (news_start_x + 20, y_pos), font, fontScale=0.6,
                                        color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow('Continents Map', canvas)

            # Check if 'q' is pressed to quit or window is closed manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Continents Map', cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
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
            ]
        }

    def fetch_feed(self, url):
        try:
            feed = feedparser.parse(url)
            return [entry.title for entry in feed.entries[:5]]
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