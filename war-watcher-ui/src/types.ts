/** General shared types used across API/client and UI. */

export type UploadsPayload = unknown; // shape is not guaranteed; we inspect flexibly

export interface CreateWarResponse {
  war: {
    war_id: number;
    my_alliance_id: number;
    enemy_alliance_id: number;
  };
  uploads: UploadsPayload;
}

export interface ProcessPayload {
  skip_rows: number[];
}

export interface ProcessResponse {
  war_id: number;
  previews: {
    latest: {
      screen_id: number;
      path: string;
      rows_detected: number;
      row_indices: number[];
      debug_overview?: string | null;
    };
    previous: null | {
      screen_id: number;
      path: string;
      rows_detected: number;
      row_indices: number[];
      debug_overview?: string | null;
    };
  };
  processing: {
    war_id: number;
    last_screen_id: number;
    attack_seq_counter_final: number;
    defence_seq_counter_final: number;
    summary: Array<{
      screen_id: number;
      rows: number;
      skipped: number[];
    }>;
  };
}
