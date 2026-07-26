from __future__ import annotations

import math
import os
import queue
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess
import chess.engine
import pygame


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets" / "pieces"
FEN_EXPORT_PATH = BASE_DIR / "fen_export.txt"
AVATAR_PATH = BASE_DIR / "ava.jpg"

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
PANEL_WIDTH = 440
WINDOW_WIDTH = BOARD_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE
FPS = 60
ANALYSIS_TIME = 0.9
COMPUTER_TIME = 0.45

LIGHT_SQUARE = (235, 220, 194)
DARK_SQUARE = (154, 111, 76)
PANEL_BG = (28, 34, 43)
PANEL_CARD = (42, 51, 63)
PANEL_BORDER = (69, 82, 98)
TEXT = (237, 242, 247)
MUTED = (165, 180, 195)
ACCENT = (93, 190, 148)
ACCENT_HOVER = (117, 210, 167)
DANGER = (205, 91, 91)
WARNING = (234, 187, 80)

PIECE_FILES = {
    "P": "Chess_plt60", "N": "Chess_nlt60", "B": "Chess_blt60",
    "R": "Chess_rlt60", "Q": "Chess_qlt60", "K": "Chess_klt60",
    "p": "Chess_pdt60", "n": "Chess_ndt60", "b": "Chess_bdt60",
    "r": "Chess_rdt60", "q": "Chess_qdt60", "k": "Chess_kdt60",
}
UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}
PALETTE_SYMBOLS = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]


def find_stockfish_path() -> str | None:
    """Find a configured Stockfish binary, including the bundled Windows build."""
    configured = os.getenv("STOCKFISH_PATH", "").strip()
    candidates = [
        configured,
        "stockfish",
        str(BASE_DIR / "stockfish"),
        str(BASE_DIR / "stockfish.exe"),
        str(BASE_DIR / "stockfish" / "stockfish"),
        str(BASE_DIR / "stockfish" / "stockfish.exe"),
        str(BASE_DIR / "stockfish" / "stockfish-windows-x86-64-avx2.exe"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).is_file():
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def read_clipboard() -> str:
    try:
        pygame.scrap.init()
        data = pygame.scrap.get(pygame.SCRAP_TEXT)
        if data:
            return data.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    except pygame.error:
        pass
    return ""


def copy_to_clipboard(text: str) -> bool:
    try:
        pygame.scrap.init()
        pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        return True
    except pygame.error:
        return False


def format_score(score: chess.engine.Score) -> str:
    if score.is_mate():
        moves = score.mate()
        return "#" if moves == 0 else f"#{moves:+d}"
    centipawns = score.score()
    return f"{(centipawns or 0) / 100:+.2f}"


def game_result_text(board: chess.Board) -> str | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is chess.WHITE:
        return "Мат. Победа белых"
    if outcome.winner is chess.BLACK:
        return "Мат. Победа чёрных"
    reasons = {
        chess.Termination.STALEMATE: "пат",
        chess.Termination.INSUFFICIENT_MATERIAL: "недостаточно материала",
        chess.Termination.SEVENTYFIVE_MOVES: "правило 75 ходов",
        chess.Termination.FIVEFOLD_REPETITION: "пятикратное повторение",
        chess.Termination.FIFTY_MOVES: "правило 50 ходов",
        chess.Termination.THREEFOLD_REPETITION: "трёхкратное повторение",
    }
    return f"Ничья: {reasons.get(outcome.termination, 'завершение партии')}"


@dataclass
class AnalysisLine:
    score: str
    pv: str
    move: chess.Move


@dataclass
class EngineJob:
    kind: Literal["analysis", "computer"]
    fen: str
    token: int
    time_limit: float


@dataclass
class EngineResult:
    kind: str
    fen: str = ""
    token: int = -1
    move: chess.Move | None = None
    lines: list[AnalysisLine] | None = None
    error: str | None = None


class EngineWorker(threading.Thread):
    """Owns Stockfish so the UI never accesses the engine concurrently."""

    def __init__(self, executable: str):
        super().__init__(daemon=True)
        self.executable = executable
        self.jobs: queue.Queue[EngineJob | None] = queue.Queue()
        self.results: queue.Queue[EngineResult] = queue.Queue()

    def submit(self, job: EngineJob) -> None:
        self.jobs.put(job)

    def stop(self) -> None:
        self.jobs.put(None)

    def run(self) -> None:
        engine: chess.engine.SimpleEngine | None = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.executable)
            self.results.put(EngineResult(kind="ready"))
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                try:
                    board = chess.Board(job.fen)
                    if job.kind == "computer":
                        result = engine.play(board, chess.engine.Limit(time=job.time_limit))
                        self.results.put(EngineResult(
                            kind="computer", fen=job.fen, token=job.token, move=result.move,
                        ))
                        continue

                    infos = engine.analyse(
                        board,
                        chess.engine.Limit(time=job.time_limit),
                        multipv=3,
                    )
                    if isinstance(infos, dict):
                        infos = [infos]
                    lines: list[AnalysisLine] = []
                    for info in infos:
                        pv = info.get("pv", [])
                        score = info.get("score")
                        if not pv or score is None:
                            continue
                        white_score = score.pov(chess.WHITE)
                        lines.append(AnalysisLine(
                            score=format_score(white_score),
                            pv=board.variation_san(pv[:8]),
                            move=pv[0],
                        ))
                    self.results.put(EngineResult(
                        kind="analysis", fen=job.fen, token=job.token, lines=lines,
                    ))
                except Exception as exc:  # Keep the interface usable after one failed job.
                    self.results.put(EngineResult(
                        kind=job.kind, fen=job.fen, token=job.token, error=str(exc),
                    ))
        except Exception as exc:
            self.results.put(EngineResult(kind="unavailable", error=str(exc)))
        finally:
            if engine is not None:
                try:
                    engine.quit()
                except Exception:
                    pass


@dataclass
class Snapshot:
    board: chess.Board
    moves: list[str]
    last_move: chess.Move | None


class ChessApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Шахматный анализатор")
        self.avatar = self.load_avatar(44)
        if self.avatar is not None:
            pygame.display.set_icon(self.avatar)
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Segoe UI", 21, bold=True)
        self.font_heading = pygame.font.SysFont("Segoe UI", 17, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 15)
        self.font_small = pygame.font.SysFont("Segoe UI", 13)
        self.font_coord = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.piece_fonts: dict[int, pygame.font.Font] = {}
        self.piece_images = self.load_piece_images()

        self.board = chess.Board()
        self.mode: Literal["play", "edit"] = "play"
        self.player_side: chess.Color | None = None  # None means local two-player game.
        self.flipped = False
        self.selected_square: chess.Square | None = None
        self.dragging_piece: tuple[chess.Square, chess.Piece] | None = None
        self.dragging_pos: tuple[int, int] | None = None
        self.selected_palette: str | None = None
        self.last_move: chess.Move | None = None
        self.moves: list[str] = []
        self.history: list[Snapshot] = []
        self.status = "Готово к игре"
        self.buttons: dict[str, pygame.Rect] = {}

        self.awaiting_promotion = False
        self.promo_from: chess.Square | None = None
        self.promo_to: chess.Square | None = None
        self.promo_color: chess.Color | None = None

        self.show_fen = False
        self.fen_editing = False
        self.fen_text = ""
        self.fen_error = ""

        self.request_token = 0
        self.engine_busy: Literal["analysis", "computer"] | None = None
        self.analysis_lines: list[AnalysisLine] = []
        self.best_move: chess.Move | None = None
        self.engine_status = "Stockfish запускается..."
        self.worker: EngineWorker | None = None
        executable = find_stockfish_path()
        if executable:
            self.worker = EngineWorker(executable)
            self.worker.start()
        else:
            self.engine_status = "Stockfish не найден"

    @staticmethod
    def load_avatar(size: int) -> pygame.Surface | None:
        try:
            avatar = pygame.image.load(str(AVATAR_PATH)).convert()
            return pygame.transform.smoothscale(avatar, (size, size))
        except (pygame.error, OSError):
            return None

    def load_piece_images(self) -> dict[str, pygame.Surface]:
        images: dict[str, pygame.Surface] = {}
        for symbol, filename in PIECE_FILES.items():
            path = ASSETS_DIR / f"{filename}.png"
            if not path.is_file():
                continue
            try:
                source = pygame.image.load(str(path)).convert_alpha()
                images[symbol] = pygame.transform.smoothscale(
                    source, (SQUARE_SIZE, SQUARE_SIZE)
                )
            except pygame.error:
                continue
        return images

    def save_snapshot(self) -> None:
        self.history.append(Snapshot(
            board=self.board.copy(stack=False),
            moves=self.moves.copy(),
            last_move=self.last_move,
        ))

    def position_changed(self) -> None:
        self.request_token += 1
        self.analysis_lines = []
        self.best_move = None

    def push_move(self, move: chess.Move, source: str = "") -> None:
        san = self.board.san(move)
        self.save_snapshot()
        self.board.push(move)
        self.moves.append(san)
        self.last_move = move
        self.selected_square = None
        self.position_changed()
        self.status = f"Ход {san}" if not source else f"Stockfish: {san}"

    def reset_game(self) -> None:
        self.board = chess.Board()
        self.history.clear()
        self.moves.clear()
        self.last_move = None
        self.selected_square = None
        self.awaiting_promotion = False
        self.dragging_piece = None
        self.dragging_pos = None
        self.show_fen = False
        self.position_changed()
        self.status = "Новая партия"
        self.maybe_request_computer_move()

    def undo(self) -> None:
        if not self.history:
            self.status = "Нет ходов для отмены"
            return
        snapshot = self.history.pop()
        self.board = snapshot.board
        self.moves = snapshot.moves
        self.last_move = snapshot.last_move
        self.selected_square = None
        self.awaiting_promotion = False
        self.position_changed()
        self.status = "Последнее действие отменено"

    def editor_changed(self) -> None:
        self.board.castling_rights = self.board.clean_castling_rights()
        self.board.ep_square = None
        self.board.halfmove_clock = 0
        self.board.clear_stack()
        self.moves.clear()
        self.last_move = None
        self.selected_square = None
        self.position_changed()

    def set_mode(self, mode: Literal["play", "edit"]) -> None:
        if mode == "play" and not self.board.is_valid():
            self.status = "Исправьте позицию в редакторе перед началом игры"
            return
        self.mode = mode
        self.selected_square = None
        self.selected_palette = None
        self.dragging_piece = None
        self.awaiting_promotion = False
        self.show_fen = False
        self.status = "Режим игры" if mode == "play" else "Режим редактора"
        self.maybe_request_computer_move()

    def cycle_player_mode(self) -> None:
        modes = [None, chess.WHITE, chess.BLACK]
        self.player_side = modes[(modes.index(self.player_side) + 1) % len(modes)]
        names = {
            None: "два игрока",
            chess.WHITE: "вы играете белыми",
            chess.BLACK: "вы играете чёрными",
        }
        self.status = f"Формат: {names[self.player_side]}"
        self.maybe_request_computer_move()

    def computer_turn(self) -> bool:
        return self.mode == "play" and self.player_side is not None and self.board.turn != self.player_side

    def maybe_request_computer_move(self) -> None:
        if (
            self.worker is None
            or self.engine_status != "Stockfish готов"
            or self.engine_busy is not None
            or not self.computer_turn()
            or not self.board.is_valid()
            or self.board.is_game_over(claim_draw=True)
            or self.awaiting_promotion
        ):
            return
        self.engine_busy = "computer"
        self.worker.submit(EngineJob("computer", self.board.fen(), self.request_token, COMPUTER_TIME))
        self.status = "Stockfish выбирает ход..."

    def request_analysis(self) -> None:
        if self.worker is None or self.engine_status != "Stockfish готов":
            self.status = self.engine_status
            return
        if self.engine_busy is not None:
            self.status = "Дождитесь завершения текущего расчёта"
            return
        if not self.board.is_valid() or self.board.is_game_over(claim_draw=True):
            self.status = "Анализ доступен только для активной корректной позиции"
            return
        self.engine_busy = "analysis"
        self.analysis_lines = []
        self.best_move = None
        self.worker.submit(EngineJob("analysis", self.board.fen(), self.request_token, ANALYSIS_TIME))
        self.status = "Stockfish анализирует позицию..."

    def process_engine_results(self) -> None:
        if self.worker is None:
            return
        try:
            while True:
                result = self.worker.results.get_nowait()
                if result.kind == "ready":
                    self.engine_status = "Stockfish готов"
                    self.status = "Stockfish готов к анализу"
                    continue
                if result.kind == "unavailable":
                    self.engine_status = f"Ошибка Stockfish: {result.error}"
                    self.status = self.engine_status
                    continue

                self.engine_busy = None
                if result.token != self.request_token or result.fen != self.board.fen():
                    continue
                if result.error:
                    self.status = f"Ошибка Stockfish: {result.error}"
                    continue
                if result.kind == "analysis":
                    self.analysis_lines = result.lines or []
                    self.best_move = self.analysis_lines[0].move if self.analysis_lines else None
                    self.status = "Анализ завершён" if self.analysis_lines else "Stockfish не вернул вариант"
                elif result.kind == "computer":
                    if result.move and self.board.is_legal(result.move):
                        self.push_move(result.move, source="engine")
                    else:
                        self.status = "Stockfish не смог выбрать легальный ход"
        except queue.Empty:
            pass
        self.maybe_request_computer_move()

    def screen_to_square(self, pos: tuple[int, int]) -> chess.Square | None:
        x, y = pos
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return None
        col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
        return chess.square(7 - col, row) if self.flipped else chess.square(col, 7 - row)

    def square_to_screen(self, square: chess.Square) -> tuple[int, int]:
        file_index, rank_index = chess.square_file(square), chess.square_rank(square)
        col, row = (7 - file_index, rank_index) if self.flipped else (file_index, 7 - rank_index)
        return col * SQUARE_SIZE, row * SQUARE_SIZE

    def square_rect(self, square: chess.Square) -> pygame.Rect:
        return pygame.Rect(*self.square_to_screen(square), SQUARE_SIZE, SQUARE_SIZE)

    def legal_targets(self) -> set[chess.Square]:
        if self.selected_square is None or not self.board.is_valid():
            return set()
        return {
            move.to_square for move in self.board.legal_moves
            if move.from_square == self.selected_square
        }

    def draw_text(
        self,
        text: str,
        pos: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int] = TEXT,
        max_width: int | None = None,
    ) -> pygame.Rect:
        if max_width is not None and font.size(text)[0] > max_width:
            shortened = text
            while shortened and font.size(shortened + "...")[0] > max_width:
                shortened = shortened[:-1]
            text = shortened + "..."
        rendered = font.render(text, True, color)
        rect = rendered.get_rect(topleft=pos)
        self.screen.blit(rendered, rect)
        return rect

    def draw_button(
        self,
        key: str,
        label: str,
        rect: pygame.Rect,
        *,
        kind: Literal["primary", "secondary", "danger"] = "secondary",
        enabled: bool = True,
    ) -> None:
        if enabled:
            self.buttons[key] = rect
        hover = rect.collidepoint(pygame.mouse.get_pos()) and enabled
        colors = {
            "primary": ACCENT_HOVER if hover else ACCENT,
            "secondary": (70, 84, 102) if hover else (57, 68, 84),
            "danger": (220, 105, 105) if hover else DANGER,
        }
        fill = colors[kind] if enabled else (54, 60, 69)
        text_color = (19, 27, 31) if kind == "primary" and enabled else TEXT if enabled else (130, 140, 150)
        pygame.draw.rect(self.screen, fill, rect, border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=4)
        label_surface = self.font_body.render(label, True, text_color)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def draw_piece(self, piece: chess.Piece, rect: pygame.Rect) -> None:
        image = self.piece_images.get(piece.symbol())
        if image:
            self.screen.blit(image, rect)
            return
        font = self.piece_fonts.get(rect.width)
        if font is None:
            font = pygame.font.SysFont("Segoe UI Symbol", int(rect.width * 0.88))
            self.piece_fonts[rect.width] = font
        glyph = UNICODE_PIECES[piece.symbol()]
        fill = (246, 246, 237) if piece.color else (29, 34, 39)
        outline = (33, 38, 43) if piece.color else (243, 238, 225)
        center = rect.center
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1)):
            outline_surface = font.render(glyph, True, outline)
            self.screen.blit(outline_surface, outline_surface.get_rect(center=(center[0] + dx, center[1] + dy - 4)))
        glyph_surface = font.render(glyph, True, fill)
        self.screen.blit(glyph_surface, glyph_surface.get_rect(center=(center[0], center[1] - 4)))

    def draw_board(self) -> None:
        targets = self.legal_targets()
        checked_king = self.board.king(self.board.turn) if self.board.is_check() else None
        for row in range(8):
            for col in range(8):
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, rect)
                square = chess.square(7 - col, row) if self.flipped else chess.square(col, 7 - row)
                if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
                    overlay.fill((246, 218, 79, 105))
                    self.screen.blit(overlay, rect)
                if square == self.selected_square:
                    pygame.draw.rect(self.screen, (87, 183, 214), rect, 4)
                if square == checked_king:
                    pygame.draw.rect(self.screen, (224, 80, 77), rect, 5)
                if square in targets:
                    if self.board.piece_at(square):
                        pygame.draw.circle(self.screen, (44, 50, 54), rect.center, SQUARE_SIZE // 2 - 7, 4)
                    else:
                        pygame.draw.circle(self.screen, (44, 50, 54), rect.center, 10)

                coordinate_color = (114, 79, 52) if color == LIGHT_SQUARE else (246, 229, 202)
                if col == 0:
                    rank = str(row + 1) if self.flipped else str(8 - row)
                    self.draw_text(rank, (rect.x + 4, rect.y + 3), self.font_coord, coordinate_color)
                if row == 7:
                    file_name = chr(ord("h") - col) if self.flipped else chr(ord("a") + col)
                    rendered = self.font_coord.render(file_name, True, coordinate_color)
                    self.screen.blit(rendered, (rect.right - rendered.get_width() - 5, rect.bottom - rendered.get_height() - 3))

        if self.best_move and self.mode == "play" and self.engine_busy != "analysis":
            self.draw_arrow(self.best_move.from_square, self.best_move.to_square)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or (self.dragging_piece and square == self.dragging_piece[0]):
                continue
            self.draw_piece(piece, self.square_rect(square))
        if self.dragging_piece and self.dragging_pos:
            rect = pygame.Rect(0, 0, SQUARE_SIZE, SQUARE_SIZE)
            rect.center = self.dragging_pos
            self.draw_piece(self.dragging_piece[1], rect)

    def draw_arrow(self, from_square: chess.Square, to_square: chess.Square) -> None:
        start = self.square_rect(from_square).center
        end = self.square_rect(to_square).center
        color = (55, 140, 224, 210)
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        pygame.draw.line(overlay, color, start, end, 8)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        wings = [
            end,
            (end[0] - 22 * math.cos(angle - math.pi / 6), end[1] - 22 * math.sin(angle - math.pi / 6)),
            (end[0] - 22 * math.cos(angle + math.pi / 6), end[1] - 22 * math.sin(angle + math.pi / 6)),
        ]
        pygame.draw.polygon(overlay, color, wings)
        self.screen.blit(overlay, (0, 0))

    def draw_promotion_dialog(self) -> None:
        if self.promo_color is None:
            return
        width, height = 4 * SQUARE_SIZE, SQUARE_SIZE
        rect = pygame.Rect((BOARD_SIZE - width) // 2, (BOARD_SIZE - height) // 2, width, height)
        shadow = rect.inflate(14, 14)
        pygame.draw.rect(self.screen, (16, 20, 25), shadow, border_radius=5)
        for index, letter in enumerate(("q", "r", "b", "n")):
            choice = pygame.Rect(rect.x + index * SQUARE_SIZE, rect.y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.screen, (224, 227, 230), choice)
            pygame.draw.rect(self.screen, (42, 48, 55), choice, 2)
            symbol = letter.upper() if self.promo_color else letter
            self.draw_piece(chess.Piece.from_symbol(symbol), choice)

    def promotion_choice_at(self, pos: tuple[int, int]) -> int | None:
        width, height = 4 * SQUARE_SIZE, SQUARE_SIZE
        rect = pygame.Rect((BOARD_SIZE - width) // 2, (BOARD_SIZE - height) // 2, width, height)
        for index, piece_type in enumerate((chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)):
            if pygame.Rect(rect.x + index * SQUARE_SIZE, rect.y, SQUARE_SIZE, SQUARE_SIZE).collidepoint(pos):
                return piece_type
        return None

    def draw_analysis(self, x: int, y: int) -> int:
        self.draw_text("АНАЛИЗ STOCKFISH", (x, y), self.font_heading)
        y += 26
        if self.engine_busy == "analysis":
            self.draw_text("Расчёт вариантов...", (x, y), self.font_small, WARNING)
            return y + 23
        if not self.analysis_lines:
            self.draw_text("Нажмите «Анализ позиции»", (x, y), self.font_small, MUTED)
            return y + 23
        for index, line in enumerate(self.analysis_lines):
            card = pygame.Rect(x, y, PANEL_WIDTH - 32, 46)
            pygame.draw.rect(self.screen, PANEL_CARD, card, border_radius=4)
            pygame.draw.rect(self.screen, PANEL_BORDER, card, 1, border_radius=4)
            self.draw_text(f"{index + 1}. {line.score}", (x + 8, y + 5), self.font_body, ACCENT)
            self.draw_text(line.pv, (x + 8, y + 25), self.font_small, TEXT, card.width - 16)
            y += 52
        self.draw_text("Оценка с точки зрения белых", (x, y), self.font_small, MUTED)
        return y + 22

    def draw_move_list(self, x: int, y: int) -> int:
        self.draw_text("ХОДЫ", (x, y), self.font_heading)
        y += 26
        if not self.moves:
            self.draw_text("Партия ещё не началась", (x, y), self.font_small, MUTED)
            return y + 21
        start = max(0, len(self.moves) - 12)
        if start % 2:
            start -= 1
        for index in range(start, len(self.moves), 2):
            move_number = index // 2 + 1
            white = self.moves[index]
            black = self.moves[index + 1] if index + 1 < len(self.moves) else ""
            self.draw_text(f"{move_number:>2}.", (x, y), self.font_small, MUTED)
            self.draw_text(white, (x + 34, y), self.font_small, TEXT, 118)
            self.draw_text(black, (x + 178, y), self.font_small, TEXT, 118)
            y += 19
        return y

    def player_mode_label(self) -> str:
        if self.player_side is chess.WHITE:
            return "Игра: вы белыми"
        if self.player_side is chess.BLACK:
            return "Игра: вы чёрными"
        return "Игра: два игрока"

    def draw_panel(self) -> None:
        self.buttons.clear()
        panel = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.line(self.screen, PANEL_BORDER, (BOARD_SIZE, 0), (BOARD_SIZE, WINDOW_HEIGHT), 1)
        x, y = BOARD_SIZE + 16, 16
        self.draw_text("ШАХМАТНАЯ ДОСКА", (x, y), self.font_title)
        self.draw_text(self.engine_status, (x, y + 30), self.font_small, ACCENT if self.engine_status == "Stockfish готов" else WARNING, PANEL_WIDTH - 32)
        if self.avatar is not None:
            avatar_rect = pygame.Rect(WINDOW_WIDTH - 60, 10, 44, 44)
            self.screen.blit(self.avatar, avatar_rect)
            pygame.draw.rect(self.screen, ACCENT, avatar_rect, 2, border_radius=4)
            self.draw_text("Dev", (WINDOW_WIDTH - 184, 15), self.font_small, MUTED)
            self.draw_text("6uxoi-Angel", (WINDOW_WIDTH - 184, 30), self.font_body, TEXT)
        y += 56

        half = (PANEL_WIDTH - 40) // 2
        self.draw_button("mode", "Игра" if self.mode == "play" else "Редактор", pygame.Rect(x, y, half, 34), kind="primary")
        self.draw_button("flip", "Перевернуть", pygame.Rect(x + half + 8, y, half, 34))
        y += 42

        if self.mode == "play":
            self.draw_button("player_mode", self.player_mode_label(), pygame.Rect(x, y, PANEL_WIDTH - 32, 34))
            y += 42
            self.draw_button("undo", "Отменить ход", pygame.Rect(x, y, half, 34), enabled=bool(self.history) and self.engine_busy is None)
            self.draw_button("new_game", "Новая партия", pygame.Rect(x + half + 8, y, half, 34), kind="danger")
            y += 42
            can_analyse = self.engine_status == "Stockfish готов" and self.engine_busy is None and self.board.is_valid() and not self.board.is_game_over(claim_draw=True)
            self.draw_button("analyse", "Анализ позиции", pygame.Rect(x, y, PANEL_WIDTH - 32, 34), kind="primary", enabled=can_analyse)
            y += 49
        else:
            self.draw_button("import", "Импорт FEN", pygame.Rect(x, y, half, 34))
            self.draw_button("export", "Экспорт FEN", pygame.Rect(x + half + 8, y, half, 34), kind="primary")
            y += 42
            self.draw_button("clear", "Очистить доску", pygame.Rect(x, y, half, 34), kind="danger")
            self.draw_button("start", "Начальная позиция", pygame.Rect(x + half + 8, y, half, 34))
            y += 43
            turn_label = "Ход белых" if self.board.turn else "Ход чёрных"
            self.draw_button("turn", turn_label, pygame.Rect(x, y, PANEL_WIDTH - 32, 32))
            y += 42
            self.draw_text("ФИГУРЫ", (x, y), self.font_heading)
            y += 25
            for index, symbol in enumerate(PALETTE_SYMBOLS):
                px = x + (index % 6) * 64
                py = y + (index // 6) * 43
                rect = pygame.Rect(px, py, 55, 36)
                selected = self.selected_palette == symbol
                pygame.draw.rect(self.screen, (87, 119, 137) if selected else PANEL_CARD, rect, border_radius=3)
                pygame.draw.rect(self.screen, ACCENT if selected else PANEL_BORDER, rect, 2 if selected else 1, border_radius=3)
                self.draw_piece(chess.Piece.from_symbol(symbol), rect)
                self.buttons[f"piece:{symbol}"] = rect
            y += 92
            self.draw_button("eraser", "Ластик", pygame.Rect(x, y, half, 32), enabled=True)
            self.draw_text("ПКМ удаляет фигуру", (x + half + 16, y + 8), self.font_small, MUTED)
            y += 46

        result = game_result_text(self.board) if self.mode == "play" and self.board.is_valid() else None
        if result:
            self.draw_text(result, (x, y), self.font_heading, DANGER if "Мат" in result else WARNING, PANEL_WIDTH - 32)
            y += 28
        elif self.mode == "play":
            turn = "Ход белых" if self.board.turn else "Ход чёрных"
            if self.computer_turn():
                turn = "Ход Stockfish"
            self.draw_text(turn, (x, y), self.font_heading, ACCENT)
            y += 28
        elif not self.board.is_valid():
            self.draw_text("Нелегальная позиция", (x, y), self.font_heading, DANGER)
            y += 28

        if self.status:
            self.draw_text(self.status, (x, y), self.font_small, MUTED, PANEL_WIDTH - 32)
            y += 25

        if self.mode == "play":
            y = self.draw_analysis(x, y) + 7
            self.draw_move_list(x, y)
        elif self.show_fen:
            self.draw_text("Введите FEN и нажмите Enter" if self.fen_editing else "FEN скопирован и сохранён", (x, y), self.font_small)
            y += 20
            field = pygame.Rect(x, y, PANEL_WIDTH - 32, 42)
            pygame.draw.rect(self.screen, (20, 25, 32), field, border_radius=3)
            pygame.draw.rect(self.screen, DANGER if self.fen_error else PANEL_BORDER, field, 1, border_radius=3)
            self.draw_text(self.fen_text, (x + 7, y + 12), self.font_small, TEXT, field.width - 14)
            if self.fen_error:
                self.draw_text(self.fen_error, (x, y + 47), self.font_small, DANGER, PANEL_WIDTH - 32)

    def draw(self) -> None:
        self.screen.fill((18, 22, 28))
        self.draw_board()
        self.draw_panel()
        if self.awaiting_promotion:
            self.draw_promotion_dialog()
        pygame.display.flip()

    def handle_board_click(self, pos: tuple[int, int], button: int) -> None:
        square = self.screen_to_square(pos)
        if square is None:
            return
        if self.mode == "edit":
            if button == 3:
                if self.board.piece_at(square):
                    self.save_snapshot()
                    self.board.remove_piece_at(square)
                    self.editor_changed()
                    self.status = "Фигура удалена"
                return
            if button != 1:
                return
            if self.selected_palette == "empty":
                if self.board.piece_at(square):
                    self.save_snapshot()
                    self.board.remove_piece_at(square)
                    self.editor_changed()
                    self.status = "Фигура удалена"
            elif self.selected_palette:
                self.save_snapshot()
                self.board.set_piece_at(square, chess.Piece.from_symbol(self.selected_palette))
                self.editor_changed()
                self.status = "Фигура добавлена"
            else:
                piece = self.board.piece_at(square)
                if piece:
                    self.save_snapshot()
                    self.dragging_piece = (square, piece)
                    self.dragging_pos = pos
                    self.board.remove_piece_at(square)
            return

        if button != 1 or self.computer_turn():
            if self.computer_turn() and button == 1:
                self.status = "Сейчас ходит Stockfish"
            return
        if not self.board.is_valid() or self.board.is_game_over(claim_draw=True):
            return
        piece = self.board.piece_at(square)
        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = square
            return
        if piece and piece.color == self.board.turn:
            self.selected_square = square
            return
        selected = self.selected_square
        moving_piece = self.board.piece_at(selected)
        if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
            promotion_moves = [
                chess.Move(selected, square, promotion=piece_type)
                for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
            ]
            if any(self.board.is_legal(move) for move in promotion_moves):
                self.awaiting_promotion = True
                self.promo_from, self.promo_to, self.promo_color = selected, square, self.board.turn
                self.selected_square = None
                return
        move = chess.Move(selected, square)
        self.selected_square = None
        if self.board.is_legal(move):
            self.push_move(move)
            self.maybe_request_computer_move()
        else:
            self.status = "Этот ход невозможен"

    def handle_button(self, key: str) -> None:
        if key == "mode":
            self.set_mode("edit" if self.mode == "play" else "play")
        elif key == "flip":
            self.flipped = not self.flipped
        elif key == "player_mode":
            self.cycle_player_mode()
        elif key == "undo":
            self.undo()
        elif key == "new_game":
            self.reset_game()
        elif key == "analyse":
            self.request_analysis()
        elif key == "import":
            self.fen_text = self.board.fen()
            self.show_fen, self.fen_editing, self.fen_error = True, True, ""
            self.status = ""
        elif key == "export":
            self.fen_text = self.board.fen()
            self.show_fen, self.fen_editing, self.fen_error = True, False, ""
            copied = copy_to_clipboard(self.fen_text)
            try:
                FEN_EXPORT_PATH.write_text(self.fen_text, encoding="utf-8")
                saved = True
            except OSError:
                saved = False
            self.status = "FEN скопирован и сохранён" if copied and saved else "FEN сохранён" if saved else "FEN показан ниже"
        elif key == "clear":
            self.save_snapshot()
            self.board.clear_board()
            self.editor_changed()
            self.status = "Доска очищена"
        elif key == "start":
            self.save_snapshot()
            self.board.reset()
            self.moves.clear()
            self.last_move = None
            self.position_changed()
            self.status = "Начальная позиция"
        elif key == "turn":
            self.save_snapshot()
            self.board.turn = not self.board.turn
            self.editor_changed()
            self.status = "Сторона хода изменена"
        elif key == "eraser":
            self.selected_palette = None if self.selected_palette == "empty" else "empty"
            self.status = "Выберите клетку для очистки" if self.selected_palette else ""
        elif key.startswith("piece:"):
            symbol = key.split(":", 1)[1]
            self.selected_palette = None if self.selected_palette == symbol else symbol
            self.status = "Выберите клетку для фигуры" if self.selected_palette else ""

    def submit_fen(self) -> None:
        try:
            new_board = chess.Board(self.fen_text.strip())
        except ValueError:
            self.fen_error = "Неверный формат FEN"
            return
        self.save_snapshot()
        self.board = new_board
        self.moves.clear()
        self.last_move = None
        self.position_changed()
        self.show_fen = self.fen_editing = False
        self.fen_error = ""
        self.status = "FEN загружен" if self.board.is_valid() else "FEN загружен: позиция требует проверки"

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.awaiting_promotion and event.button == 1:
                choice = self.promotion_choice_at(event.pos)
                if choice and self.promo_from is not None and self.promo_to is not None:
                    move = chess.Move(self.promo_from, self.promo_to, promotion=choice)
                    if self.board.is_legal(move):
                        self.push_move(move)
                        self.maybe_request_computer_move()
                    self.awaiting_promotion = False
                    self.promo_from = self.promo_to = self.promo_color = None
                return True
            if event.pos[0] < BOARD_SIZE:
                self.handle_board_click(event.pos, event.button)
            elif event.button == 1:
                for key, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self.handle_button(key)
                        break
        elif event.type == pygame.MOUSEMOTION and self.dragging_piece:
            self.dragging_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging_piece:
            source, piece = self.dragging_piece
            target = self.screen_to_square(event.pos)
            if target is None:
                self.board.set_piece_at(source, piece)
                self.history.pop()  # The drag never changed the position.
            else:
                self.board.set_piece_at(target, piece)
                self.editor_changed()
                self.status = "Фигура перемещена"
            self.dragging_piece = self.dragging_pos = None
        elif event.type == pygame.KEYDOWN:
            if self.awaiting_promotion and event.key == pygame.K_ESCAPE:
                self.awaiting_promotion = False
                self.promo_from = self.promo_to = self.promo_color = None
            elif self.show_fen and self.fen_editing:
                if event.key == pygame.K_RETURN:
                    self.submit_fen()
                elif event.key == pygame.K_ESCAPE:
                    self.show_fen = self.fen_editing = False
                    self.fen_error = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.fen_text = self.fen_text[:-1]
                elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                    self.fen_text = (self.fen_text + read_clipboard())[:180]
                elif event.unicode and event.unicode.isprintable():
                    self.fen_text = (self.fen_text + event.unicode)[:180]
            elif event.key == pygame.K_ESCAPE:
                self.selected_square = None
            elif event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
                self.undo()
            elif event.key == pygame.K_f:
                self.flipped = not self.flipped
        return True

    def run(self) -> None:
        running = True
        while running:
            self.process_engine_results()
            self.draw()
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break
            self.clock.tick(FPS)
        self.shutdown()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=2)
        pygame.quit()


def main() -> None:
    ChessApp().run()


if __name__ == "__main__":
    main()
