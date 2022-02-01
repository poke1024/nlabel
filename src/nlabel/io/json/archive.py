class Archive:
    def _collections(self, progress=False, **kwargs):
        raise NotImplementedError()

    def save_to_qda(self, path, *selectors, exist_ok=False, progress=True):
        from .qda import Exporter as QDAExporter

        exporter = QDAExporter(path, *selectors, exist_ok=exist_ok)
        with exporter.writer() as writer:
            for _, doc in self._collections(progress=progress):
                writer.add(doc)
