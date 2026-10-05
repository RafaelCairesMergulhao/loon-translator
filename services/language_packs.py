from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from services.translation_engine import LANGUAGES_DICT

logger = logging.getLogger(__name__)


class LanguagePackError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LanguagePackStatus:
    name: str
    code: str
    installed: bool


class LanguagePackManager:
    """Gerencia pares Argos locais usando inglês como idioma de interligação."""

    LAUNCH_LANGUAGES = (
        "Português",
        "Inglês",
        "Espanhol",
        "Francês",
        "Alemão",
        "Italiano",
        "Hindi",
        "Chinês (Mandarim)",
        "Japonês",
        "Coreano",
        "Russo",
        "Árabe",
    )

    @staticmethod
    def _package_module():
        try:
            import argostranslate.package

            return argostranslate.package
        except ImportError as exc:
            raise LanguagePackError(
                "O mecanismo Argos Translate não está instalado nesta edição."
            ) from exc

    def statuses(self) -> list[LanguagePackStatus]:
        installed_pairs = self._installed_pairs()
        result: list[LanguagePackStatus] = []
        for name in self.LAUNCH_LANGUAGES:
            code = LANGUAGES_DICT[name].lower().split("-")[0]
            result.append(LanguagePackStatus(name, code, self._is_installed(code, installed_pairs)))
        return result

    def missing(self, language_names: list[str]) -> list[str]:
        installed_pairs = self._installed_pairs()
        missing: list[str] = []
        for name in language_names:
            if name in {"", "Auto"}:
                continue
            code = LANGUAGES_DICT.get(name, "").lower().split("-")[0]
            if not code:
                continue
            if not self._is_installed(code, installed_pairs) and name not in missing:
                missing.append(name)
        return missing

    def _installed_pairs(self) -> set[tuple[str, str]]:
        package = self._package_module()
        return {
            (item.from_code.split("_")[0], item.to_code.split("_")[0])
            for item in package.get_installed_packages()
        }

    @staticmethod
    def _is_installed(code: str, installed_pairs: set[tuple[str, str]]) -> bool:
        return code == "en" or ((code, "en") in installed_pairs and ("en", code) in installed_pairs)

    def install(self, language_name: str, progress=None) -> None:
        package = self._package_module()
        code = LANGUAGES_DICT.get(language_name, "").lower().split("-")[0]
        if not code or code == "auto":
            raise LanguagePackError(f"Idioma inválido: {language_name}")
        if code == "en":
            return

        if progress:
            progress("Atualizando catálogo de pacotes...")
        package.update_package_index()
        available = package.get_available_packages()
        installed_pairs = {
            (item.from_code.split("_")[0], item.to_code.split("_")[0])
            for item in package.get_installed_packages()
        }
        for source, target in ((code, "en"), ("en", code)):
            if (source, target) in installed_pairs:
                continue
            candidates = [
                item
                for item in available
                if item.from_code.split("_")[0] == source
                and item.to_code.split("_")[0] == target
            ]
            if not candidates:
                raise LanguagePackError(
                    f"Não há pacote local disponível para {source} → {target}."
                )
            selected = candidates[-1]
            if progress:
                progress(f"Baixando {source} → {target}...")
            downloaded: Path = selected.download()
            if progress:
                progress(f"Instalando {source} → {target}...")
            package.install_from_path(downloaded)
        if progress:
            progress(f"{language_name} instalado para uso offline.")
