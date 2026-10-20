from common.config import SvcSettings
from common.logging_helper import configure_logger

from . import create_app

settings = SvcSettings()
configure_logger(settings.log_level, settings.log_format)
app = create_app()
