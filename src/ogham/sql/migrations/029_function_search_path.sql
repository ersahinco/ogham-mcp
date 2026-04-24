-- src/ogham/sql/migrations/029_function_search_path.sql
--
-- Harden trigger functions from migrations 026 + 028 against the
-- "Function Search Path Mutable" warning Supabase raises. See the
-- canonical copy under sql/migrations/029_function_search_path.sql
-- for the rationale; this is the tree-sync'd duplicate.

BEGIN;

CREATE OR REPLACE FUNCTION init_memory_lifecycle() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    INSERT INTO memory_lifecycle (memory_id, profile, stage, stage_entered_at, updated_at)
    VALUES (NEW.id, NEW.profile, 'fresh', NEW.created_at, NEW.created_at)
    ON CONFLICT (memory_id) DO NOTHING;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION sync_memory_lifecycle_profile() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    IF NEW.profile IS DISTINCT FROM OLD.profile THEN
        UPDATE memory_lifecycle
           SET profile = NEW.profile, updated_at = now()
         WHERE memory_id = NEW.id;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION topic_summaries_set_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

COMMIT;
