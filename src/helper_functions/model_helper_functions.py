from loguru import logger

"""
Jag tror nog denna egentligen är lite onödig då den inte gör något mer än att kalla på funktioner
med i princip samma namn som de har.
Men vill du ha kvar den ändå så bör du döpa den till något mer beskrivande för vad klassen gör, dvs något med loggning
Funktionerna bör också vara staticmethod eller classmethod då de inte använder sig av några instansvariabler
"""


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
