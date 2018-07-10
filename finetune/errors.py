class InvalidTargetType(ValueError):
    def __init__(self, target_type, *args, **kwargs):
        super().__init__(self, "Invalid target_type: {}".format(target_type))
