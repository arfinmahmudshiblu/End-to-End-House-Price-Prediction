# from src.logger import logging

# logging.info("Logging setup complete.")

from src.exception import CustomException
import sys
def test_exception():
    try:
        a=1/0
    except Exception as e:
        raise CustomException(e,sys)
    