import cv2
import numpy as np
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from gaze_detection import AdvancedGazeMovementControl
import os
import mediapipe as mp
import pyautogui
import util
from pynput.mouse import Button, Controller

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class ContinentMapApp:
    def __init__(self, image_path, new_width=1500):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        original_height, original_width = self.image.shape[:2]
        aspect_ratio = original_height / original_width
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.draw = mp.solutions.drawing_utils
        self.mouse = Controller()
        self.screen_width = new_width
        self.screen_width, self.screen_height = pyautogui.size()
        self.map_width = int(self.screen_width * 0.75)  
        self.news_width = int(self.screen_width * 0.25)  
        new_height = int(self.map_width * aspect_ratio)
        self.resized_image = cv2.resize(self.image, (self.map_width, new_height))
        self.current_continent = None
        self.displayed_continent = None
        self.news_fetcher = ContinentNewsFetcher()
        self.news_headlines = {}
        self.preload_news()
        self.scroll_position = 0
        self.max_scroll = 0
        self.gaze_control = AdvancedGazeMovementControl()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture device")
        
    def preload_news(self):
        print("Preloading news for all continents...")
        continents = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Australia']
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_continent = {executor.submit(self.news_fetcher.get_news_for_continent, continent): continent for continent in continents}
            for future in as_completed(future_to_continent):
                continent = future_to_continent[future]
                try:
                    self.news_headlines[continent] = future.result()
                    print(f"News fetched for {continent}")
                except Exception as exc:
                    print(f"Error fetching news for {continent}: {exc}")
        print("News preloading complete.")

    def get_continent_colors(self):
        return {
            'North America': [203, 192, 255],  
            'South America': [0, 255, 255],    
            'Europe': [255, 204, 153],         
            'Africa': [159, 166, 6],           
            'Asia': [0, 204, 0],               
            'Australia': [153, 102, 51],       
            #' ': [255, 255, 255]      # White (BGR)
        }

    def check_continent_by_color(self, x, y):
        if x >= self.map_width or y >= self.resized_image.shape[0]:
            return None
        pixel_color = self.resized_image[y, x].tolist()
        continent_colors = self.get_continent_colors()
        closest_continent = None
        min_diff = float('inf')
        for continent_name, color in continent_colors.items():
            if continent_name != 'Antarctica':  # Exclude Antarctica from color matching
                diff = np.linalg.norm(np.array(pixel_color) - np.array(color))
                if diff < min_diff:
                    min_diff = diff
                    closest_continent = continent_name
        return closest_continent
    
    def find_finger_tip(self, processed):
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
            return index_finger_tip
        return None

    def move_mouse(self, index_finger_tip):
        if index_finger_tip is not None:
            x = int((1 - index_finger_tip.x) * self.screen_width)
            y = int(index_finger_tip.y * self.screen_height)
            pyautogui.moveTo(x, y)

    def is_left_click(self, landmark_list, thumb_index_dist):
        return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
        )

    def all_fingers_open(self, landmark_list, thumb_index_dist):
        return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 60 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 60 and
            thumb_index_dist > 50
        )

    def all_fingers_closed(self, landmark_list):
        return all(
            util.get_distance([landmark_list[finger_tip], landmark_list[finger_pip]]) > 30
            for finger_tip, finger_pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
        ) and util.get_distance([landmark_list[4], landmark_list[3]]) > 30

    def detect_hand_gesture(self, frame, landmark_list, processed):
        if len(landmark_list) >= 21:
            index_finger_tip = self.find_finger_tip(processed)
            corrected_landmarks = [(1 - x, y) for x, y in landmark_list]
            thumb_index_dist = util.get_distance([corrected_landmarks[4], corrected_landmarks[8]])

            if util.get_angle(corrected_landmarks[5], corrected_landmarks[6], corrected_landmarks[8]) > 170:
                self.move_mouse(index_finger_tip)
            elif self.is_left_click(landmark_list, thumb_index_dist):
                self.mouse.press(Button.left)
                self.mouse.release(Button.left)
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self.all_fingers_open(landmark_list, thumb_index_dist):
                pyautogui.scroll(20)
                cv2.putText(frame, "Scroll Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self.all_fingers_closed(landmark_list):
                pyautogui.scroll(-20)
                cv2.putText(frame, "Scroll Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_continent = self.check_continent_by_color(x, y)
            if clicked_continent and clicked_continent != 'Antarctica':
                self.current_continent = clicked_continent
                self.displayed_continent = clicked_continent
                self.scroll_position = 0
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scroll_position = max(0, self.scroll_position - 50)
            else:
                self.scroll_position = min(self.max_scroll, self.scroll_position + 50)
    
    def wrap_text(self, text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = font.getlength(word)
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + font.getlength(' ')
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width + font.getlength(' ')
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def run(self):
        cv2.namedWindow('Continents Map', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Continents Map', self.mouse_event)
        if not self.cap.isOpened():
            print("Error: Camera not accessible")
            return
        canvas_height = int(self.screen_height * 0.9)  
        canvas_width = self.screen_width   
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        title_font = ImageFont.truetype("ARIAL.ttf", 36)
        content_font = ImageFont.truetype("ARIAL.ttf", 20)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.gaze_control.process_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.hands.process(frame_rgb)
            
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                self.draw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((1 - lm.x, lm.y))
                
                self.detect_hand_gesture(frame, landmark_list, processed)
            canvas = np.ones((self.resized_image.shape[0], self.screen_width, 3), dtype=np.uint8) * 255
            scale = min(canvas_height / self.resized_image.shape[0], 
                    (canvas_width * 0.75) / self.resized_image.shape[1])
            map_display_width = int(self.resized_image.shape[1] * scale)
            map_display_height = int(self.resized_image.shape[0] * scale)
            displayed_map = cv2.resize(self.resized_image, (map_display_width, map_display_height))
            canvas[:map_display_height, :map_display_width] = displayed_map
            news_start_x = self.map_width + 10
            news_height = min(canvas.shape[0] - 40, self.resized_image.shape[0] - 40)  
            news_width = self.news_width - 20
            news_image = Image.new('RGB', (news_width, news_height), color=(240, 240, 240))
            draw = ImageDraw.Draw(news_image)
            if self.current_continent:
                draw.text((20, 20), self.current_continent, font=title_font, fill=(0, 0, 0))
            if self.displayed_continent:
                y_offset = 65 - self.scroll_position
                line_spacing = 30
                headline_spacing = 10
                for headline in self.news_headlines[self.displayed_continent]:
                    wrapped_text = self.wrap_text(headline, content_font, news_width - 40)
                    if y_offset + len(wrapped_text) * line_spacing > 65:
                        if y_offset >= 65:
                            draw.ellipse([20, y_offset + 8, 28, y_offset + 16], fill=(0, 0, 0))
                        for line in wrapped_text:
                            if y_offset >= 65:
                                draw.text((40, y_offset), line, font=content_font, fill=(0, 0, 0))
                            y_offset += line_spacing
                    else:
                        y_offset += len(wrapped_text) * line_spacing
                    y_offset += headline_spacing
                self.max_scroll = max(0, y_offset - news_height)
            news_array = np.array(news_image)
            if news_array.shape[0] > news_height:
                news_array = news_array[:news_height, :, :]
            canvas[20:20+news_height, news_start_x:news_start_x+news_width] = news_array
            cv2.imshow('Continents Map', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Continents Map', cv2.WND_PROP_VISIBLE) < 1:
                break
        self.cap.release()
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
            ],
            'Antarctica': [
                'https://antarcticsun.usap.gov/feed/'
            ]
        }

    def fetch_feed(self, url):
        try:
            feed = feedparser.parse(url)
            return [entry.title for entry in feed.entries[:20]]  
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