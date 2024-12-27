from enum import Enum, unique

@unique
class ErrorMessage(Enum):
    NOT_FOUND = '리소스를 찾지 못했습니다'
