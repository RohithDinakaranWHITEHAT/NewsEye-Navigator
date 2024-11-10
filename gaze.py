import cv2
import numpy as np
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import dlib
import time

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
        self.news_fetcher = ContinentNewsFetcher()
        self.news_headlines = {}
        self.fetch_all_news()
        
        # Initialize gaze detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("H:/cvip project/NewsEye-Navigator/shape_predictor_68_face_landmarks.dat")
        self.last_gaze_time = time.time()
        self.gaze_duration = 1.0  # Duration in seconds to trigger continent selection

    def fetch_all_news(self):
        print("Fetching news for all continents...")
        for continent in self.get_continent_colors().keys():
            self.news_headlines[continent] = self.news_fetcher.get_news_for_continent(continent)
        print("News fetching complete.")

    def get_continent_colors(self):
        return {
            'North America': [203, 192, 255],  # Pink (BGR)
            'South America': [0, 255, 255],    # Yellow (BGR)
            'Europe': [255, 204, 153],         # Light Blue (BGR)
            'Africa': [170, 255, 195],         # Teal/Cyan-green (BGR)
            'Asia': [0, 204, 0],               # Green (BGR)
            'Australia': [153, 102, 51],       # Blue (BGR)
            'Antarctica': [255, 255, 255]      # White (BGR)
        }

    def check_continent_by_color(self, x, y):
        """
        Check which continent is clicked based on the pixel color at (x, y).
        """
        if x < 0 or x >= self.resized_image.shape[1] or y < 0 or y >= self.resized_image.shape[0]:
            return None
        
        pixel_color = self.resized_image[y, x].tolist()  # Get BGR color at (x, y)
        continent_colors = self.get_continent_colors()
        closest_continent = None
        min_diff = float('inf')
        
        for continent_name, color in continent_colors.items():
            diff = np.linalg.norm(np.array(pixel_color) - np.array(color))  # Euclidean distance between colors
            if diff < min_diff:
                min_diff = diff
                closest_continent = continent_name
        
        return closest_continent
    def detect_gaze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Define eye landmarks
            left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(39).x, landmarks.part(39).y)])
            right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(45).x, landmarks.part(45).y)])
            
            # Calculate gaze ratio
            left_eye_points = [36, 37, 38, 39, 40, 41]
            right_eye_points = [42, 43, 44, 45, 46, 47]
            gaze_ratio = self.get_gaze_ratio(gray, landmarks, left_eye_points, right_eye_points)
            
            # Determine gaze direction
            if gaze_ratio <= 1:
                gaze_x = int(self.map_width * (1 - gaze_ratio))
            else:
                gaze_x = int(self.map_width * (2 - gaze_ratio))
            
            gaze_y = (left_eye[0][1] + right_eye[0][1]) // 2
            
            # Scale gaze coordinates to match resized image dimensions
            scaled_gaze_x = int(gaze_x * (self.resized_image.shape[1] / frame.shape[1]))
            scaled_gaze_y = int(gaze_y * (self.resized_image.shape[0] / frame.shape[0]))
            
            # Ensure coordinates are within bounds
            scaled_gaze_x = max(0, min(scaled_gaze_x, self.resized_image.shape[1] - 1))
            scaled_gaze_y = max(0, min(scaled_gaze_y, self.resized_image.shape[0] - 1))
            
            # Check which continent is being gazed at
            continent = self.check_continent_by_color(scaled_gaze_x, scaled_gaze_y)
            
            if continent:
                self.current_continent = continent
            
            # Draw a circle to indicate gaze point (on the original frame)
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
        
        return frame

    def get_gaze_ratio(self, gray, landmarks, left_eye_points, right_eye_points):
        # Get the left eye region
        left_eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in left_eye_points])
        
        # Create a mask for the left eye
        height, width = gray.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        
        # Extract the eye from the grayscale image
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Define the minimum enclosing rectangle for the eye region
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        
        # Extract the eye image
        gray_eye = eye[min_y: max_y, min_x: max_x]
        
        # Threshold the eye region
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        
        # Calculate the gaze ratio
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_white = cv2.countNonZero(right_side_threshold)
        
        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        
        return gaze_ratio
    def run(self):
        cv2.namedWindow('H:/cvip project/NewsEye-Navigator/Continents.png', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_with_gaze = self.detect_gaze(frame)
            
            # Create a blank canvas for the entire screen
            canvas = np.ones((self.resized_image.shape[0], self.screen_width, 3), dtype=np.uint8) * 255
            
            # Place the map on the left side
            canvas[:, :self.map_width] = self.resized_image
            
            # Draw the news section on the right side
            news_start_x = self.map_width + 10
            cv2.rectangle(canvas, (news_start_x, 10), (news_start_x + self.news_width - 10, canvas.shape[0] - 10), (240, 240, 240), -1)
            
            if self.current_continent:
                font_scale, font_thickness, font = 1.2, 2, cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, self.current_continent, (news_start_x + 20, 60), font, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness, lineType=cv2.LINE_AA)
                
                line_y_offset = 120
                max_lines = 5
                for i, headline in enumerate(self.news_headlines[self.current_continent][:max_lines]):
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
                            cv2.putText(canvas, f"{i+1}. {line}" if j == 0 else f" {line}", (news_start_x + 20, y_pos), font, fontScale=0.6, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            
            # Display the gaze detection frame
            gaze_frame_height = int(frame_with_gaze.shape[0] * (self.news_width / frame_with_gaze.shape[1]))
            resized_gaze_frame = cv2.resize(frame_with_gaze, (self.news_width, gaze_frame_height))
            canvas[canvas.shape[0] - gaze_frame_height:, self.map_width:] = resized_gaze_frame
            
            cv2.imshow('H:/cvip project/NewsEye-Navigator/Continents.png', canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if cv2.getWindowProperty('H:/cvip project/NewsEye-Navigator/Continents.png', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cap.release()
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
    app = ContinentMapApp('H:/cvip project/NewsEye-Navigator/Continents.png')
    app.run()