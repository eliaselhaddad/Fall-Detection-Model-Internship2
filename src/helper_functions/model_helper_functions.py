from loguru import logger


class ModelHelpingFunctions:
    def __init__(self):
        pass

    def log_info(self, message):
        logger.info(message)

    def log_warning(self, message):
        logger.warning(message)

    def log_error(self, message):
        logger.error(message)

    def log_success(self, message):
        logger.success(message)

    def log_exception(self, message):
        logger.exception(message)

    def log_info(self, message):
        logger.info(message)
