<script>
    export let imname
    export let thr = -10
    export let source_server

</script>
<style>


</style>

<div>
	<slot>
        <img src="{source_server}/marked_image/{imname}" alt="{imname}"><br>
    </slot>
</div>