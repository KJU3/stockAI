from abc import *

class training_model_template_class(metaclass = ABCMeta):

    # 템플릿 메서드
    def training_model_template_method(self):
        # 데이터 불러오기
        data = self.load_data()

        # x y 데이터 나누기
        x_y_data = self.split_x_y_data(data)

        # 훈련 테스트 데이터 나누기
        train_and_test_data = self.split_train_and_test_data(x_y_data)

        # 데이터 전처리
        preprocessing_data = self.preprocess_data(train_and_test_data)

        # 모델 훈련
        model_and_score = self.trainig_model(preprocessing_data)

        # 전처리 데이터 저장
        self.save_preprocessing_data(preprocessing_data)

        # 모델 저장
        self.save_model(model_and_score)
        return
    
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def split_x_y_data(self, data):
        pass
    
    @abstractmethod
    def split_train_and_test_data(self, data):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass
    
    @abstractmethod
    def trainig_model(self, data):
        pass

    @abstractmethod
    def save_preprocessing_data(self, data):
        pass

    @abstractmethod
    def save_model(self, data):
        pass