from werkzeug.routing import BaseConverter


class ListConverter(BaseConverter):

    def to_python(self, value):
        return value.split(',')

    def to_url(self, value):
        base_to_url = super(ListConverter, self).to_url
        return ','.join(base_to_url(v) for v in value)
