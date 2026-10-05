import os
import logging

logger = logging.getLogger("FirebaseClient")

class FirebaseDatabaseClient:
    """Cliente real para persistência no Firebase Realtime Database."""
    def __init__(self, database_url: str, credential_path: str = "serviceAccountKey.json"):
        self.database_url = database_url
        self.is_simulated = True
        self._db = None
        self._initialize_firebase(credential_path)

    def _initialize_firebase(self, credential_path: str):
        try:
            import firebase_admin
            from firebase_admin import credentials, db

            self._db = db
            if not os.path.exists(credential_path):
                logger.warning("Credenciais Firebase ausentes; integração desativada.")
                return
            try:
                firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(credential_path)
                firebase_admin.initialize_app(cred, {"databaseURL": self.database_url})
            self.is_simulated = False
        except (ImportError, OSError, ValueError) as e:
            logger.warning("Firebase desativado: %s", e)
            self.is_simulated = True

    def save_translation_record(self, record_dict: dict):
        """Salva o registo de tradução diretamente no nó '/history' do Realtime Database."""
        try:
            if not self.is_simulated:
                ref = self._db.reference('loontranslator-history')
                ref.push(record_dict)
                logger.info(f"[Firebase Sync] Registo gravado com sucesso na nuvem em {record_dict['timestamp']}")
            else:
                logger.debug("[Firebase desativado] Registo ignorado.")
        except Exception as e:
            logger.error(f"Erro ao sincronizar com o Firebase: {e}")