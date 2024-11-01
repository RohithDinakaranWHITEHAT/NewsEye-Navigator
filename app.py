import cv2
import numpy as np
from newsdataapi import NewsDataApiClient

class ContinentMapApp:
    def __init__(self, image_path, api_key, new_width=1500):
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        height, width = self.image.shape[:2]

        aspect_ratio = height / width
        new_height = int(new_width * aspect_ratio)

        self.resized_image = cv2.resize(self.image, (new_width, new_height))

        self.continents = {
            'North America': (50, 100, 400, 300),   
            'South America': (370, 450, 550, 720),  
            'Europe': (700, 120, 850, 320),       
            'Africa': (650, 350, 800, 600),         
            'Asia': (850, 100, 1150, 400),        
            'Australia': (1100, 500, 1250, 650),    
            'Antarctica': (400, 700, 1300, 800)     
        }

        self.current_continent = None

        self.news_fetcher = ContinentNewsFetcher(api_key)

        self.news_headlines = []

    def check_continent(self, x, y):
        for continent_name, (x1, y1, x2, y2) in self.continents.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return continent_name
        return None

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            continent = self.check_continent(x, y)
            if continent:
                self.current_continent = continent
                self.news_headlines = self.news_fetcher.get_news_for_continent(continent)
            else:
                self.current_continent = None

    def run(self):
        cv2.namedWindow('Continents Map', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Continents Map', self.mouse_event)

        while True:
            img_copy = self.resized_image.copy()  
            box_width = 320
            box_height = 320
            cv2.rectangle(img_copy,
                          (10 ,img_copy.shape[0] - box_height -10),
                          (10 +box_width ,img_copy.shape[0] -10),
                          (255 ,255 ,255), -1)   

            cv2.rectangle(img_copy,
                          (10 ,img_copy.shape[0] - box_height -10),
                          (10 +box_width ,img_copy.shape[0] -10),
                          (0 ,0 ,0), 2)   

            if self.current_continent:
                cv2.putText(img_copy,
                            self.current_continent,
                            (20 ,img_copy.shape[0] - box_height +30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0,
                            color=(0 ,0 ,0),   
                            thickness=2,
                            lineType=cv2.LINE_AA)

                line_y_offset = img_copy.shape[0] - box_height +70

                for i ,headline in enumerate(self.news_headlines[:10]):
                    cv2.putText(img_copy,
                                f"{i+1}. {headline}",
                                (20,line_y_offset +(i *30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.3,
                                color=(0 ,0 ,0),
                                thickness=1,
                                lineType=cv2.LINE_AA)

            cv2.imshow('Continents Map', img_copy)

            if cv2.waitKey(1) &0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

class ContinentNewsFetcher:
    def __init__(self, api_key):
        self.newsapi = NewsDataApiClient(apikey=api_key)

    def get_news_for_continent(self, continent_name):
        continent_to_query = {
            'North America': 'us',  
            'South America': 'br', 
            'Europe': 'gb',         
            'Africa': 'za',          
            'Asia': 'cn',            
            'Australia': 'au',       
            'Antarctica': None       
        }

        country_code = continent_to_query.get(continent_name)

        if country_code:
            response = self.newsapi.news_api(country=country_code, language='en')
        else:
            return ["No news available for Antarctica"]
        articles = response.get('results', [])
        headlines = [article['title'] for article in articles]

        return headlines if headlines else ["No news found"]

if __name__ == "__main__":
    app = ContinentMapApp('D:/Fall 2024/Computer vision/NewsEye-Navigator/Continents.png',api_key='pub_57878d93ffb64b6a846cbb538669468d5c96b')
    app.run()

