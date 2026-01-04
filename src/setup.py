"""
CogniDoc Interactive Setup Wizard.

Provides a guided setup experience:
1. Configure LLM and embedding providers
2. Validate API keys and services
3. Detect and process documents
4. Launch the chat interface
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint
from dotenv import load_dotenv, set_key

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# Custom questionary style
custom_style = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'bold'),
    ('answer', 'fg:cyan'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
    ('separator', 'fg:gray'),
    ('instruction', 'fg:gray'),
])

# Provider configurations
PROVIDERS = {
    "ollama": {
        "name": "Ollama (local, gratuit)",
        "requires_key": False,
        "llm_models": ["granite3.3:8b", "llama3.2:8b", "qwen2.5:7b", "mistral:7b"],
        "vision_models": ["qwen3-vl:8b-instruct", "llava:13b", "llava:7b"],
        "embed_models": ["qwen3-embedding:0.6b", "nomic-embed-text", "mxbai-embed-large"],
    },
    "gemini": {
        "name": "Google Gemini",
        "requires_key": True,
        "key_env": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        "llm_models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "vision_models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "cost_per_1k_tokens": 0.000075,  # Approximate for flash
    },
    "openai": {
        "name": "OpenAI",
        "requires_key": True,
        "key_env": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "llm_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "vision_models": ["gpt-4o", "gpt-4o-mini"],
        "embed_models": ["text-embedding-3-small", "text-embedding-3-large"],
        "cost_per_1k_tokens": 0.0025,  # Approximate for gpt-4o-mini
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "requires_key": True,
        "key_env": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/settings/keys",
        "llm_models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
        "vision_models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
        "cost_per_1k_tokens": 0.003,  # Approximate for sonnet
    },
}

# Required Ollama models for document processing
REQUIRED_OLLAMA_MODELS = [
    ("ibm/granite-docling:258m-bf16", "Document parsing (DocLing)"),
]


class SetupWizard:
    """Interactive setup wizard for CogniDoc."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.env_path = self.base_dir / ".env"
        self.pdf_dir = self.base_dir / "data" / "pdfs"
        self.config = {}

        # Ensure directories exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Load existing env
        load_dotenv(self.env_path)

    def print_header(self):
        """Print the wizard header."""
        console.print()
        console.print(Panel.fit(
            "[bold cyan]CogniDoc Setup Wizard[/bold cyan]\n"
            "[dim]Configuration interactive pour votre assistant documentaire[/dim]",
            border_style="cyan",
        ))
        console.print()

    def print_step(self, step: int, total: int, title: str):
        """Print a step header."""
        console.print()
        console.print(f"[bold cyan][{step}/{total}] {title}[/bold cyan]")
        console.print("[dim]" + "─" * 50 + "[/dim]")

    # =========================================================================
    # Step 1: LLM Provider Configuration
    # =========================================================================

    def configure_llm_provider(self) -> dict:
        """Configure the LLM provider for generation."""
        self.print_step(1, 4, "Configuration du LLM")

        # Select provider
        choices = [
            questionary.Choice(
                title=f"{PROVIDERS[p]['name']}",
                value=p
            )
            for p in ["ollama", "gemini", "openai", "anthropic"]
        ]

        provider = questionary.select(
            "Quel provider pour la génération de texte ?",
            choices=choices,
            style=custom_style,
        ).ask()

        if provider is None:
            raise KeyboardInterrupt()

        provider_config = PROVIDERS[provider]
        config = {"provider": provider}

        # Get API key if required
        if provider_config["requires_key"]:
            key_env = provider_config["key_env"]
            existing_key = os.getenv(key_env)

            if existing_key:
                use_existing = questionary.confirm(
                    f"Clé API {key_env} détectée. L'utiliser ?",
                    default=True,
                    style=custom_style,
                ).ask()

                if use_existing:
                    config["api_key"] = existing_key
                else:
                    config["api_key"] = self._ask_api_key(provider, key_env)
            else:
                console.print(f"[dim]Obtenez votre clé sur: {provider_config['key_url']}[/dim]")
                config["api_key"] = self._ask_api_key(provider, key_env)

        # Select model
        models = provider_config["llm_models"]
        model = questionary.select(
            "Quel modèle utiliser ?",
            choices=models,
            default=models[0],
            style=custom_style,
        ).ask()

        config["model"] = model

        # Validate connection
        if not self._validate_provider(provider, config):
            console.print("[red]Échec de la validation. Veuillez vérifier vos paramètres.[/red]")
            return self.configure_llm_provider()

        console.print(f"[green]✓[/green] LLM configuré: {provider} / {model}")
        return config

    def _ask_api_key(self, provider: str, key_env: str) -> str:
        """Ask for an API key."""
        key = questionary.password(
            f"Entrez votre clé API {provider}:",
            style=custom_style,
        ).ask()

        if not key:
            raise ValueError("Clé API requise")

        return key

    def _validate_provider(self, provider: str, config: dict) -> bool:
        """Validate that the provider works."""
        console.print("[dim]Validation de la connexion...[/dim]")

        try:
            if provider == "ollama":
                return self._validate_ollama()
            elif provider == "gemini":
                return self._validate_gemini(config.get("api_key"))
            elif provider == "openai":
                return self._validate_openai(config.get("api_key"))
            elif provider == "anthropic":
                return self._validate_anthropic(config.get("api_key"))
        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")
            return False

        return True

    def _validate_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            import ollama
            client = ollama.Client()
            client.list()
            console.print("[green]✓[/green] Ollama connecté")
            return True
        except Exception as e:
            console.print(f"[red]✗ Ollama non disponible: {e}[/red]")
            console.print("[yellow]Lancez Ollama avec: ollama serve[/yellow]")
            return False

    def _validate_gemini(self, api_key: str) -> bool:
        """Validate Gemini API key."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            model.generate_content("test", generation_config={"max_output_tokens": 5})
            console.print("[green]✓[/green] Clé API Gemini validée")
            return True
        except Exception as e:
            console.print(f"[red]✗ Erreur Gemini: {e}[/red]")
            return False

    def _validate_openai(self, api_key: str) -> bool:
        """Validate OpenAI API key."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.models.list()
            console.print("[green]✓[/green] Clé API OpenAI validée")
            return True
        except Exception as e:
            console.print(f"[red]✗ Erreur OpenAI: {e}[/red]")
            return False

    def _validate_anthropic(self, api_key: str) -> bool:
        """Validate Anthropic API key."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Just check auth works
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            console.print("[green]✓[/green] Clé API Anthropic validée")
            return True
        except Exception as e:
            console.print(f"[red]✗ Erreur Anthropic: {e}[/red]")
            return False

    # =========================================================================
    # Step 2: Embedding Provider Configuration
    # =========================================================================

    def configure_embeddings(self) -> dict:
        """Configure the embedding provider."""
        self.print_step(2, 4, "Configuration des Embeddings")

        # For embeddings, Ollama is strongly recommended (free + fast)
        choices = [
            questionary.Choice(
                title="Ollama - qwen3-embedding:0.6b (recommandé, gratuit)",
                value=("ollama", "qwen3-embedding:0.6b"),
            ),
            questionary.Choice(
                title="Ollama - nomic-embed-text",
                value=("ollama", "nomic-embed-text"),
            ),
            questionary.Choice(
                title="OpenAI - text-embedding-3-small",
                value=("openai", "text-embedding-3-small"),
            ),
        ]

        provider, model = questionary.select(
            "Quel provider pour les embeddings ?",
            choices=choices,
            style=custom_style,
        ).ask()

        config = {"provider": provider, "model": model}

        # If OpenAI, need API key
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY") or self._ask_api_key("openai", "OPENAI_API_KEY")
            config["api_key"] = key

        console.print(f"[green]✓[/green] Embeddings configurés: {provider} / {model}")
        return config

    # =========================================================================
    # Step 3: Ollama Models Verification
    # =========================================================================

    def verify_ollama_models(self, llm_config: dict, embed_config: dict) -> bool:
        """Verify and download required Ollama models."""
        self.print_step(3, 4, "Vérification des modèles Ollama")

        # Collect required models
        required_models = []

        # DocLing is always required
        for model, desc in REQUIRED_OLLAMA_MODELS:
            required_models.append((model, desc))

        # Add LLM model if using Ollama
        if llm_config["provider"] == "ollama":
            required_models.append((llm_config["model"], "LLM génération"))

        # Add embedding model if using Ollama
        if embed_config["provider"] == "ollama":
            required_models.append((embed_config["model"], "Embeddings"))

        if not required_models:
            console.print("[dim]Aucun modèle Ollama requis[/dim]")
            return True

        # Check Ollama connection
        try:
            import ollama
            client = ollama.Client()
            available = {m["name"].split(":")[0]: m["name"] for m in client.list()["models"]}
        except Exception as e:
            console.print(f"[red]Impossible de se connecter à Ollama: {e}[/red]")
            console.print("[yellow]Lancez Ollama avec: ollama serve[/yellow]")
            return False

        # Check each model
        missing = []
        table = Table(title="Modèles Ollama")
        table.add_column("Modèle", style="cyan")
        table.add_column("Usage", style="dim")
        table.add_column("Status")

        for model, desc in required_models:
            model_base = model.split(":")[0]
            if model_base in available or model in [m["name"] for m in client.list()["models"]]:
                table.add_row(model, desc, "[green]✓ Disponible[/green]")
            else:
                table.add_row(model, desc, "[red]✗ Manquant[/red]")
                missing.append(model)

        console.print(table)

        # Download missing models
        if missing:
            download = questionary.confirm(
                f"Télécharger les {len(missing)} modèle(s) manquant(s) ?",
                default=True,
                style=custom_style,
            ).ask()

            if download:
                for model in missing:
                    console.print(f"[dim]Téléchargement de {model}...[/dim]")
                    try:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(),
                            TaskProgressColumn(),
                            console=console,
                        ) as progress:
                            task = progress.add_task(f"Downloading {model}", total=None)

                            # Use subprocess to show progress
                            process = subprocess.Popen(
                                ["ollama", "pull", model],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )

                            for line in process.stdout:
                                if "%" in line:
                                    progress.update(task, description=line.strip())

                            process.wait()

                            if process.returncode == 0:
                                console.print(f"[green]✓[/green] {model} téléchargé")
                            else:
                                console.print(f"[red]✗[/red] Échec du téléchargement de {model}")
                                return False
                    except Exception as e:
                        console.print(f"[red]Erreur: {e}[/red]")
                        return False
            else:
                console.print("[yellow]Les modèles manquants sont requis pour le traitement[/yellow]")
                return False

        console.print("[green]✓[/green] Tous les modèles sont disponibles")
        return True

    # =========================================================================
    # Step 4: Save Configuration
    # =========================================================================

    def save_configuration(self, llm_config: dict, embed_config: dict):
        """Save configuration to .env file."""
        self.print_step(4, 4, "Sauvegarde de la configuration")

        env_vars = {
            "DEFAULT_LLM_PROVIDER": llm_config["provider"],
            "DEFAULT_VISION_PROVIDER": llm_config["provider"],
        }

        # LLM model
        if llm_config["provider"] == "ollama":
            env_vars["OLLAMA_LLM_MODEL"] = llm_config["model"]
            env_vars["OLLAMA_VISION_MODEL"] = "qwen3-vl:8b-instruct"
        else:
            env_vars["DEFAULT_LLM_MODEL"] = llm_config["model"]
            env_vars["DEFAULT_VISION_MODEL"] = llm_config["model"]

        # API keys
        if llm_config.get("api_key"):
            key_env = PROVIDERS[llm_config["provider"]]["key_env"]
            env_vars[key_env] = llm_config["api_key"]

        # Embeddings
        env_vars["EMBED_MODEL"] = embed_config["model"]
        if embed_config["provider"] == "openai" and embed_config.get("api_key"):
            env_vars["OPENAI_API_KEY"] = embed_config["api_key"]

        # Save to .env
        for key, value in env_vars.items():
            set_key(str(self.env_path), key, value)

        console.print(f"[green]✓[/green] Configuration sauvegardée dans {self.env_path}")

        # Show summary
        table = Table(title="Configuration")
        table.add_column("Paramètre", style="cyan")
        table.add_column("Valeur")

        table.add_row("LLM Provider", llm_config["provider"])
        table.add_row("LLM Model", llm_config["model"])
        table.add_row("Embed Provider", embed_config["provider"])
        table.add_row("Embed Model", embed_config["model"])

        console.print(table)

    # =========================================================================
    # Document Processing
    # =========================================================================

    def detect_documents(self) -> list[Path]:
        """Detect PDF documents in the data directory."""
        pdfs = list(self.pdf_dir.glob("*.pdf"))
        return pdfs

    def show_documents(self, pdfs: list[Path]) -> tuple[list[Path], dict]:
        """Show detected documents and get user confirmation."""
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Traitement de documents[/bold cyan]",
            border_style="cyan",
        ))
        console.print()

        if not pdfs:
            console.print(f"[yellow]Aucun document trouvé dans {self.pdf_dir}[/yellow]")
            console.print(f"[dim]Ajoutez vos fichiers PDF dans ce dossier et relancez le wizard[/dim]")
            return [], {}

        # Show documents
        table = Table(title=f"Documents dans {self.pdf_dir}")
        table.add_column("#", style="dim")
        table.add_column("Fichier", style="cyan")
        table.add_column("Taille")

        total_size = 0
        for i, pdf in enumerate(pdfs, 1):
            size = pdf.stat().st_size
            total_size += size
            size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.0f} KB"
            table.add_row(str(i), pdf.name, size_str)

        console.print(table)

        # Estimate time and cost
        estimates = self._estimate_processing(pdfs, total_size)

        console.print()
        console.print(f"[bold]Estimation:[/bold]")
        console.print(f"  [dim]Temps:[/dim] ~{estimates['time_minutes']:.0f} minutes")
        if estimates['cost'] > 0:
            console.print(f"  [dim]Coût API:[/dim] ~${estimates['cost']:.2f}")

        return pdfs, estimates

    def _estimate_processing(self, pdfs: list[Path], total_size: int) -> dict:
        """Estimate processing time and cost."""
        # Rough estimates based on file size
        # ~1 page per 50KB, ~30 seconds per page for full pipeline
        estimated_pages = total_size / (50 * 1024)
        time_minutes = estimated_pages * 0.5  # 30 seconds per page

        # Cost estimate if using paid API
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
        cost = 0
        if provider in PROVIDERS and "cost_per_1k_tokens" in PROVIDERS[provider]:
            # Rough estimate: ~1000 tokens per page
            cost = estimated_pages * 1 * PROVIDERS[provider]["cost_per_1k_tokens"]

        return {
            "time_minutes": max(time_minutes, 1),
            "cost": cost,
            "estimated_pages": estimated_pages,
        }

    def process_documents(self, pdfs: list[Path]) -> bool:
        """Run the ingestion pipeline on documents."""
        if not pdfs:
            return False

        proceed = questionary.confirm(
            f"Traiter ces {len(pdfs)} document(s) ?",
            default=True,
            style=custom_style,
        ).ask()

        if not proceed:
            return False

        console.print()
        console.print("[bold cyan]Lancement du pipeline d'ingestion...[/bold cyan]")
        console.print()

        # Import and run pipeline
        try:
            from .run_ingestion_pipeline import run_ingestion_pipeline_async

            # Run with progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Pipeline en cours...", total=10)

                # Run the pipeline
                async def run_with_progress():
                    stats = await run_ingestion_pipeline_async(
                        vision_provider=os.getenv("DEFAULT_VISION_PROVIDER", "ollama"),
                        skip_pdf=False,
                        skip_yolo=False,
                        skip_extraction=False,
                        skip_descriptions=False,
                        skip_chunking=False,
                        skip_embeddings=False,
                        skip_indexing=False,
                        skip_graph=False,
                    )
                    return stats

                # Run the async pipeline
                stats = asyncio.run(run_with_progress())
                progress.update(task, completed=10)

            console.print()
            console.print("[green]✓ Pipeline terminé avec succès![/green]")

            # Show stats
            if stats.get("embeddings"):
                console.print(f"  [dim]Chunks créés:[/dim] {stats['embeddings'].get('total_chunks', 'N/A')}")
            if stats.get("graph_building"):
                gb = stats["graph_building"]
                console.print(f"  [dim]Entités extraites:[/dim] {gb.get('total_nodes', 'N/A')}")
                console.print(f"  [dim]Relations:[/dim] {gb.get('total_edges', 'N/A')}")

            return True

        except Exception as e:
            console.print(f"[red]Erreur pendant le traitement: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False

    # =========================================================================
    # Action Menu
    # =========================================================================

    def show_action_menu(self):
        """Show the action menu after processing."""
        while True:
            console.print()
            console.print(Panel.fit(
                "[bold cyan]Que faire ensuite ?[/bold cyan]",
                border_style="cyan",
            ))

            action = questionary.select(
                "Choisissez une action:",
                choices=[
                    questionary.Choice(title="Lancer CogniDoc (interface web)", value="launch"),
                    questionary.Choice(title="Ajouter d'autres documents", value="add_docs"),
                    questionary.Choice(title="Relancer le traitement", value="reprocess"),
                    questionary.Choice(title="Quitter", value="quit"),
                ],
                style=custom_style,
            ).ask()

            if action == "launch":
                self.launch_webapp()
                break
            elif action == "add_docs":
                console.print(f"\n[cyan]Ajoutez vos PDFs dans:[/cyan] {self.pdf_dir}")
                console.print("[dim]Puis sélectionnez 'Relancer le traitement'[/dim]")
                questionary.press_any_key_to_continue(style=custom_style).ask()
            elif action == "reprocess":
                pdfs = self.detect_documents()
                if pdfs:
                    pdfs, _ = self.show_documents(pdfs)
                    self.process_documents(pdfs)
                else:
                    console.print(f"[yellow]Aucun document dans {self.pdf_dir}[/yellow]")
            elif action == "quit":
                console.print("\n[dim]Au revoir![/dim]")
                break

    def launch_webapp(self):
        """Launch the CogniDoc web application."""
        console.print()
        console.print("[bold cyan]Lancement de CogniDoc...[/bold cyan]")
        console.print("[dim]L'interface sera disponible sur http://localhost:7860[/dim]")
        console.print("[dim]Appuyez sur Ctrl+C pour arrêter[/dim]")
        console.print()

        try:
            from .cognidoc_app import main as run_app
            run_app()
        except KeyboardInterrupt:
            console.print("\n[dim]Application arrêtée[/dim]")

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(self):
        """Run the setup wizard."""
        try:
            self.print_header()

            # Check if already configured
            if self.env_path.exists():
                existing_provider = os.getenv("DEFAULT_LLM_PROVIDER")
                if existing_provider:
                    reconfigure = questionary.confirm(
                        f"Configuration existante détectée ({existing_provider}). Reconfigurer ?",
                        default=False,
                        style=custom_style,
                    ).ask()

                    if not reconfigure:
                        # Skip to document processing
                        pdfs = self.detect_documents()
                        if pdfs:
                            pdfs, _ = self.show_documents(pdfs)
                            self.process_documents(pdfs)
                        self.show_action_menu()
                        return

            # Step 1: Configure LLM
            llm_config = self.configure_llm_provider()

            # Step 2: Configure embeddings
            embed_config = self.configure_embeddings()

            # Step 3: Verify Ollama models
            if not self.verify_ollama_models(llm_config, embed_config):
                console.print("[red]Configuration incomplète. Veuillez réessayer.[/red]")
                return

            # Step 4: Save configuration
            self.save_configuration(llm_config, embed_config)

            # Document processing
            pdfs = self.detect_documents()
            if pdfs:
                pdfs, _ = self.show_documents(pdfs)
                self.process_documents(pdfs)
            else:
                console.print()
                console.print(f"[yellow]Ajoutez vos PDFs dans {self.pdf_dir} pour les traiter[/yellow]")

            # Action menu
            self.show_action_menu()

        except KeyboardInterrupt:
            console.print("\n[dim]Configuration annulée[/dim]")
        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    """Main entry point."""
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    main()
